import math
import copy
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from torch.nn import Dropout, Softmax, Linear, Conv1d, LayerNorm, functional
from models import net_config as configs
from os.path import join
import logging
from models.net_resnet_skip import ResNetV2
from models.sspcab import SSPCAB
# from timm.models.registry import register_model

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

CONFIGS = {
    'BASELINE': configs.get_b16_config(),
    'TEST': configs.get_test_config(),
    'Res50_b16': configs.get_r50_b16_config(),
    'BASELINE12': configs.get_b16_tr12_config(),
    'TEST_RES': configs.get_r50_test_config()
}


# convert HWIO( LIO ) to OIHW ( OIL )
def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([2, 1, 0])
    return torch.from_numpy(weights)


class Embeddings(nn.Module):
    def __init__(self, config, signal_size, in_channels=32):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config

        if config.patches.get("grid") is not None:  # 混合结构
            grid_size = config.patches["grid"]  # grid = 128
            patch_size = signal_size // 16 // grid_size  # 4096 / 16 / 256 = 1
            n_patches = int(signal_size / (patch_size * 16))  # 4096 / (1 * 16) = 256
            self.hybrid = True
        else:
            patch_size = config.patches["size"]  # 16
            n_patches = int(signal_size / patch_size)  # 
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16  # width=64  64 * 16 = 1024 resnet50最后输出的通道数
        self.patch_embedding = Conv1d(in_channels=in_channels, out_channels=config.hidden_size, kernel_size=patch_size,
                                      stride=patch_size)  # kernel_size == stride 不重叠的获取
        # 添加位置信息
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))  # [1,256(n_patches),512]
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:  # 混合模型
            x, features = self.hybrid_model(x)  # x -> torch.Size([1, 1024, 256])
        else:
            features = None
        x = x.to(torch.float32)  # weight是floatTensor类型的
        x = self.patch_embedding(x)  # Conv1d ： [B, C, L] 
        x = x.transpose(-1, 1)  # [1,512,256] -> [1,256,512] 

        embeddings = x + self.position_embeddings  # [1,256,512] + [1,n_patches,hidden] add 
        embeddings = self.dropout(embeddings)  # [B, 256(n_patches), 512]
        return embeddings, features


class Attention(nn.Module):
    def __init__(self, config, is_mask):
        super(Attention, self).__init__()
        self.config = config
        self.is_mask = is_mask
        self.attention_heads = config.transformer["num_heads"]  # 4
        self.attention_head_size = int(config.hidden_size / self.attention_heads)  # 512 / 4 = 128
        self.all_head_size = self.attention_heads * self.attention_head_size  # 128*4=512

        self.query = Linear(config.hidden_size, self.all_head_size)  # Linear(512,512) 更像是（256，128）
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)

        self.atten_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_score(self, x):  # 
        x_new_shape = x.size()[:-1] + (self.attention_heads, self.attention_head_size)
        x = x.view(*x_new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_state):
        mixed_query_layer = self.query(hidden_state)
        mixed_key_layer = self.key(hidden_state)
        mixed_value_layer = self.value(hidden_state)

        query_layer = self.transpose_for_score(mixed_query_layer)
        key_layer = self.transpose_for_score(mixed_key_layer)
        value_layer = self.transpose_for_score(mixed_value_layer)

        attention_score = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_score)
        weights = attention_probs if self.is_mask else None
        attention_probs = self.atten_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*context_new_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.activate_func = functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate_func(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransBlock(nn.Module):
    def __init__(self, config, is_mask):
        super(TransBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention = Attention(config, is_mask)
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attention(x)
        x += h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x += h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # np2th : convert HWIO（ LIO ） to OIHW  （OIL）
            query_weight = np2th(weights[join(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                  self.hidden_size).t()
            key_weight = np2th(weights[join(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[join(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                  self.hidden_size).t()
            out_weight = np2th(weights[join(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                  self.hidden_size).t()

            query_bias = np2th(weights[join(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[join(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[join(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[join(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attention.query.weight.copy_(query_weight)
            self.attention.key.weight.copy_(key_weight)
            self.attention.value.weight.copy_(value_weight)
            self.attention.out.weight.copy_(out_weight)
            self.attention.query.bias.copy_(query_bias)
            self.attention.key.bias.copy_(key_bias)
            self.attention.value.bias.copy_(value_bias)
            self.attention.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[join(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[join(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[join(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[join(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[join(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[join(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[join(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[join(ROOT, MLP_NORM, "bias")]))


# n层transformer + layernorm
class Encoder(nn.Module):
    def __init__(self, config, is_mask):
        super(Encoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList()
        self.is_mask = is_mask
        self.encode_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = TransBlock(config, is_mask)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attention_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.is_mask:
                attention_weights.append(weights)
        encoded = self.encode_norm(hidden_states)
        return encoded, attention_weights


# embedding+transformer层
class Transformer(nn.Module):
    def __init__(self, config, signal_size, is_mask):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, signal_size=signal_size)
        self.encoder = Encoder(config, is_mask)

    def forward(self, input_c):  # input_c [4096]
        embedding_out, features = self.embeddings(input_c)
        encoded, attention_w = self.encoder(embedding_out)
        return encoded, attention_w, features


class Conv1dRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batch_norm=False):
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not use_batch_norm)
        relu = nn.ReLU(inplace=True)
        # bn = LayerNorm(out_channels)  # 修改成LN
        bn = nn.BatchNorm1d(out_channels)  # original
        super(Conv1dRelu, self).__init__(conv, bn, relu)


# 上采样块
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batch_norm=False):
        super(DecodeBlock, self).__init__()
        self.conv1 = Conv1dRelu(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batch_norm=use_batch_norm)
        self.conv2 = Conv1dRelu(out_channels, out_channels, kernel_size=3, padding=1,
                                use_batch_norm=use_batch_norm)
        self.up = nn.Upsample(scale_factor=2)  # 上采样两倍，即L变为原来的两倍

    def forward(self, x, skip=None):
        x = self.up(x)  # 注意这里 L 变为原来的两倍  
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  #  【B,  ,256】
  
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.drop_2(x)
        return x


# 解码器
class Decoder(nn.Module):
    def __init__(self, config, add_sspcab=False):
        super(Decoder, self).__init__()
        self.config = config
        self.add_sspcab = add_sspcab
        head_channels = 512

        self.sspcab = SSPCAB(head_channels)  # new add
        self.mse_loss = nn.MSELoss()

        self.conv_relu = Conv1dRelu(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batch_norm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels  # skip_channels = [512, 256, 64, 16]
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0  # skip_channels [512, 256, 64, 16] -> [512, 256, 64, 0]
        else:
            skip_channels = [0] * 4  # [0,0,0,0]

        blocks = [
            DecodeBlock(in_channel, out_channel, skip_channel) for in_channel, out_channel, skip_channel in
            zip(in_channels, out_channels, skip_channels)
            # in : [512, 256, 128, 64]  out : [256, 128, 64, 16]  skip : [512, 256, 64, 0]
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        x = hidden_states.permute(0, 2, 1).contiguous()  # -> [B, C, L] = [B, hidden_size, n_patches]
        x = self.conv_relu(x)  # -> [head_channels(512), n_patches(256)]
        hidden_map = x
        cost_sspcab = 0
        if self.add_sspcab:
            output_sspcab = self.sspcab(x)
            cost_sspcab = self.mse_loss(output_sspcab, x)
            x = output_sspcab

        # 解码块组
        for i, decode_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None  # n_skip=3
            else:
                skip = None
            x = decode_block(x, skip=skip)  # skip依次取features 的前三个
        return x, hidden_map, cost_sspcab


class FinalClass(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, add_sim=False):
        super(FinalClass, self).__init__()
        self.in_channels = in_channels  # 
        self.kernel_size = kernel_size
        self.out_channels = out_channels  # = config['n_classes']
        self.add_sim = add_sim
        self.conv1d = Conv1d(self.in_channels, out_channels, kernel_size=self.kernel_size,
                             padding=self.kernel_size // 2)
        self.fc = Linear(4096, 4096)
        # self.fc = nn.Sequential(
        #     Linear(in_channels, in_channels * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(in_channels * 4, out_channels),
        # )

    def forward(self, x):
        x = self.conv1d(x)  # base:[1,16,L] -> [1,1,L(4096)] 
        if self.add_sim:
            return x
        b, c, l = x.size()
        x = x.view(b, c * l)
        x = self.fc(x)
        # x=x.unsqueeze(1)
        return x


# 主体网络
class TransUnet(nn.Module):
    def __init__(self, config, signal_size=4096, num_classes=1, is_mask=False, add_sspcab=False, add_sim=False):
        super(TransUnet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.add_sspcab = add_sspcab
        self.add_sim = add_sim
        self.classifier = config.classifier  # seg or token(一般用于test)
        self.transformer = Transformer(config, signal_size, is_mask)
        self.decoder = Decoder(config, add_sspcab=add_sspcab)
        self.final_class = FinalClass(in_channels=config['decoder_channels'][-1],
                                      out_channels=config['n_classes'], kernel_size=3, add_sim=add_sim)

    def forward(self, x):
        x = x.repeat(1, 32, 1)
        x, attention_w, features = self.transformer(x)  # 这里features是混合模型输出的feature map
        x, hidden_map, cost_sspcab = self.decoder(x, features)
        final_tensor = self.final_class(x)
        if self.add_sim:
            # print(f"final tensor size = {final_tensor.size()}") # [B, n_cls, L(4096)]
            return final_tensor
        return final_tensor, hidden_map, cost_sspcab

    def load_from(self, weights):
        with torch.no_grad():
            res_weights = weights
            self.transformer.embeddings.patch_embedding.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embedding.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encode_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encode_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            pos_emb = np2th(weights["Transformer/pos_embed_input/pos_embedding"])
            pos_emb_new = self.transformer.embeddings.position_embeddings

            if pos_emb.size() == pos_emb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(pos_emb)
            elif pos_emb.size()[1] - 1 == pos_emb_new.size()[1]:
                pos_emb = pos_emb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(pos_emb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (pos_emb.size(), pos_emb_new.size()))
                n_tok_new = pos_emb_new.size(1)  #
                if self.classifier == "seg":
                    _, pos_emb_grid = pos_emb[:, :1], pos_emb[0, 1:]
                gs_old = int(len(pos_emb_grid))
                gs_new = int(n_tok_new)
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                pos_emb_grid = pos_emb_grid.reshape(gs_old, -1)
                seq_zoom = (gs_new / gs_old, 1)
                pos_emb_grid = zoom(pos_emb_grid, seq_zoom, order=1)  # th2np
                pos_emb_grid = pos_emb_grid.reshape(1, gs_new, -1)
                pos_emb = pos_emb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(pos_emb))

            # Encoder 共享权重
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:  # 混合模型
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weights, n_block=bname, n_unit=uname)

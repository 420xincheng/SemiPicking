"""
original code :
https://github1s.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/HEAD/pytorch_classification/ConvNeXt/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
        The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
        shape (batch_size, height, width, channels) while channels_first corresponds to inputs
        with shape (batch_size, channels, height, width).
        """

    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        # 初始化偏置和权重
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":  # 直接用LN，LN是对C这个维度进行归一化
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [B, C, H, W]  对batch维度归一化
            mean = x.mean(1, keepdim=True)  # 均值： dim=1, keepdim:保持维度不变
            var = (x - mean).pow(2).mean(1, keepdim=True)  # var方差
            x = (x - mean) / torch.sqrt(var + self.eps)  # BN
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

       This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
       the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
       See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
       changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
       'survival rate' as the argument.

       """
    if drop_prob == 0. or not training:  # 不在训练（测试/验证的时候直接返回）
        return x
    keep_prob = 1 - drop_prob  # 保留率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # x.ndim 返回x的维度   (1, ) * (x.ndim - 1) => (x.ndim - 1)个1  即(1,1,1,..,)
    # shape = (x.shape[0], 1, 1, ...)  x.dim-1个1
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # x.dtype 返回数据类型
    random_tensor.floor_()  # 向下取整
    output = x.div(keep_prob) * random_tensor  # x.div  张量和标量之间做除法
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
        We use (2) as we find it slightly faster in PyTorch

        Args:
            dim (int): Number of input channels.
            drop_rate (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # group=dim，深度卷积，每个通道都用一个卷积核处理。
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 1×1卷积（pointwise conv 点卷积） 作用相当于Linear
        self.act = nn.GELU()  # 激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)  # pointwise conv 相当于 Linear
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C] 将通道数移到最后一个位置，之后在进行LN
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 1, 2)  # [B, L, C] -> [B, C, L] 将通道数的位置还原回来

        x = shortcut + self.drop_path(x)  # 加上原来的输入
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
            A PyTorch impl of : `A ConvNet for the 2020s`  -
              https://arxiv.org/pdf/2201.03545.pdf
        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """

    def __init__(self, in_chans=3, num_classes=1000, depths: list = None, dims: list = None,
                 drop_path_rate=0, layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # 下采样组
        stem = nn.Sequential(nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format='channels_first'))  # 最开始的输入图像之后部分 L = L / 4
        self.downsample_layers.append(stem)  # 放到下采样layers的第一个

        # 对应后续阶段2-4的三个downsample , 在ConvNeXt Block之前的downsample
        for i in range(3):
            # downsample由一个LN和一个kernel_size=stride=2的卷积组成(H,W 都减半)   dims Default: [96, 192, 384, 768]
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4个特征解析stage， 每个阶段都由不同的块组成
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # torch.linspace(start, end, steps) 将end-st分为steps  块堆叠的次数：depths-Default: [3, 3, 9, 3]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )  # * 表示
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # 最后一层norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))   # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnext_base(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model
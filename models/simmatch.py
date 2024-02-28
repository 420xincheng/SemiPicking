# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import TransUnet, CONFIGS


class ResNet(nn.Module):
    def __init__(self, base_encoder, num_classes, norm_layer=None, dim=128):
        super(ResNet, self).__init__()
        self.trainMode = base_encoder
        if base_encoder == 'BASELINE':
            print(base_encoder)
            self.backbone = TransUnet(CONFIGS[base_encoder], signal_size=num_classes, num_classes=num_classes,
                                      add_sspcab=True, add_sim=True)
            assert not hasattr(self.backbone, 'fc'), "fc should not in backbone"
            self.fc = nn.Linear(CONFIGS[base_encoder]['n_classes'], 1)
            self.head = nn.Sequential(
                nn.Linear(4096, CONFIGS[base_encoder]['n_classes']),
                nn.ReLU(inplace=True),
                nn.Linear(CONFIGS[base_encoder]['n_classes'], dim),
            )
        else:
            self.backbone = base_encoder(norm_layer=norm_layer)
            self.fc = nn.Linear(self.backbone.out_channels, num_classes)
            self.head = nn.Sequential(
                nn.Linear(self.backbone.out_channels, self.backbone.out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.backbone.out_channels, dim),
            )

    def forward(self, x):
        if self.trainMode == "BASELINE":
            x = self.backbone(x)  # [B, C(1), L(4096)]
            embedding = self.head(x)
            logits = self.fc(x.permute(0, 2, 1))
            return logits.squeeze(-1), F.normalize(embedding.squeeze(1))
        else:
            x = x.repeat(1, 32, 1)
            x = self.backbone(x)
            embedding = self.head(x)
            logits = self.fc(x)
            return logits, F.normalize(embedding)


'''
dim：embedding 的维度大小，默认为128。
K：bank 的大小，即 SimMatch 类中存储的队列的大小，默认为256。
'''

class SimMatch(nn.Module):
    def __init__(self, base_encoder, num_classes=4096, eman=False, momentum=0.999, dim=128, norm=None, K=256,
                 args=None):
        super(SimMatch, self).__init__()
        self.eman = eman
        self.momentum = momentum
        self.num_classes = num_classes
        self.main = ResNet(base_encoder, num_classes, norm_layer=norm, dim=dim)
        # build ema models
        self.device = torch.device(f"cuda:{str(args.gpu)}" if torch.cuda.is_available() else "cpu")
        print("using EMAN as techer models")
        self.ema = ResNet(base_encoder, num_classes, norm_layer=norm, dim=dim)
        for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
            param_ema.data.copy_(param_main.data)  # initialize
            param_ema.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("bank", torch.randn(dim, K))
        self.bank = nn.functional.normalize(self.bank, dim=0)  # 存储了K个最近的样本特征表示的循环缓冲区
        self.register_buffer("labels", torch.zeros(K, dtype=torch.long))

        if args.DA:
            self.DA_len = 256
            self.register_buffer("DA_queue", torch.zeros(self.DA_len, num_classes, dtype=torch.float))  # 存储数据增强后的样本特征表示
            self.register_buffer("DA_ptr", torch.zeros(1, dtype=torch.long))

    def momentum_update_ema(self):
        if self.eman:
            state_dict_main = self.main.state_dict()
            state_dict_ema = self.ema.state_dict()
            for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
                assert k_main == k_ema, "state_dict names are different!"
                assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
                if 'num_batches_tracked' in k_ema:
                    v_ema.copy_(v_main)
                else:
                    v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
        else:
            for param_q, param_k in zip(self.main.parameters(), self.ema.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _update_bank(self, k, labels, index):
        self.bank[:, index] = k.t()
        self.labels[index] = labels.squeeze(-1)

    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.mean(0)
        torch.distributed.all_reduce(probs_bt_mean)
        ptr = int(self.DA_ptr)
        self.DA_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()
        self.DA_ptr[0] = (ptr + 1) % self.DA_len
        probs = probs / self.DA_queue.mean(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) models. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) models. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    """
    im_x:有标签数据的输入
    im_u_w, im_u_s: 无标签的输入, 分别代表 weak 和 strong 数据；
    labels 和 index：有标签数据的类别和对应的索引；
    start_unlabel：标志位，表示是否开始无标签数据的训练；
    
    output:
        logits_qx：有标签数据的分类输出；
        prob_ku：无标签数据的预测概率；
        logits_qu：强无标签数据的分类输出；
        loss_in：无标签数据的 loss。
    """

    def forward(self, im_x, im_u_w=None, im_u_s=None, labels=None, index=None, start_unlabel=False, args=None):
        if im_u_w is None and im_u_s is None:
            logits, _ = self.main(im_x)
            return logits

        batch_x = im_x.shape[0]
        batch_u = im_u_w.shape[0]
        bank = self.bank.clone().detach()
        # im_x = [B,1, 4096] im_u = [arg.mu(5) * B, 1, 4096]
        logits_q, feat_q = self.main(torch.cat((im_x, im_u_s)))
        logits_qx, logits_qu = logits_q[:batch_x], logits_q[batch_x:]
        feat_qx, feat_qu = feat_q[:batch_x], feat_q[batch_x:]
        with torch.no_grad():
            im_k = torch.cat([im_x, im_u_w])
            if self.eman:
                logits_k, feat_k = self.ema(im_k)
                print(f'model line 170 logits_k {logits_k.size()}, {feat_k.size()}')
            else:
                # im, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                logits_k, feat_k = self.ema(im_k)
                # logits_k, feat_k = self.ema(im)
                # feat_k = self._batch_unshuffle_ddp(feat_k, idx_unshuffle)
                # logits_k = self._batch_unshuffle_ddp(logits_k, idx_unshuffle)

            _, logits_ku = logits_k[:batch_x], logits_k[batch_x:]
            feat_kx, feat_ku = feat_k[:batch_x], feat_k[batch_x:]
            prob_ku_orig = F.softmax(logits_ku.float(), dim=-1)
            if args.DA:
                prob_ku_orig = self.distribution_alignment(prob_ku_orig)

        if start_unlabel:
            with torch.no_grad():
                teacher_logits = feat_ku @ bank
                teacher_prob_orig = F.softmax(teacher_logits / args.tt, dim=1)
                factor = prob_ku_orig.gather(1, self.labels.expand([batch_u, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if args.c_smooth < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels.expand([bs, -1]), teacher_prob_orig)
                    prob_ku = prob_ku_orig * args.c_smooth + aggregated_prob * (1 - args.c_smooth)
                else:
                    prob_ku = prob_ku_orig

            student_logits = feat_qu @ bank
            student_prob = F.softmax(student_logits / args.st, dim=1)
            loss_in = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1)
        else:
            loss_in = torch.tensor(0, dtype=torch.float).to(self.device)
            prob_ku = prob_ku_orig
        self._update_bank(feat_kx, labels, index)
        return logits_qx, prob_ku, logits_qu, loss_in


def get_simmatch_model(model):
    if isinstance(model, str):
        model = {
            "SimMatch": SimMatch,
        }[model]
    return model


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output

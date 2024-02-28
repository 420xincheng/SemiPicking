#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import LayerNorm, GroupNorm, InstanceNorm1d

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, signal_size=4096, light=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc  # 输入通道数
        self.output_nc = output_nc  # 输出通道数
        self.ngf = ngf  # 第一层卷积后通道数
        self.n_blocks = n_blocks  # 残差块数
        self.signal_size = signal_size  # 信号size
        self.light = light  # 是否使用轻量化模型

        DownBlock1 = []
        DownBlock1 += [nn.ReflectionPad1d(3),  # 7*7卷积核，Padding为3
                       nn.Conv1d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                       nn.InstanceNorm1d(ngf),  # IN归一化
                       nn.LeakyReLU(0.2, True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock1 += [nn.ReflectionPad1d(1),
                           nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                           nn.InstanceNorm1d(ngf * mult * 2),
                           nn.LeakyReLU(0.2, True)]

        DownBlock2 = []
        # Down-Sampling Bottleneck(编码器残差块) * 6
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock2 += [ResnetBlock(ngf * mult, use_bias=False)]
            # 四个残差块不改变通道数，均为256->256

        # Class Activation Map（类别激活图CAM）
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)  # global average pooling后的全连接层
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)  # global max pooling后的全连接层
        # 将global average pooling特征图与global max pooling特征图融合叠加在一起，通道数变为256+256=512

        self.conv1x1 = nn.Conv1d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        # 将叠加后的特征图降维，512->256
        self.relu = nn.LeakyReLU(0.2, True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.LeakyReLU(0.2, True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.LeakyReLU(0.2, True)]
        else:
            FC = [nn.Linear(signal_size // mult * signal_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.LeakyReLU(0.2, True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.LeakyReLU(0.2, True)]
        # 轻量化：256->256, 256->256, 非轻量化signal_size/4*signal_size/4*64*4(signal_size^2/16)->256

        # AdaILN中的Gamma和Beta
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
        # 均为256->256

        # Up-Sampling Bottleneck(解码器残差块) * 6
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))
            # 设置属性值，self.UpBlock1_i，为AdaILN归一化（可以改成GN或者LN试一试）

        # Up-Sampling（上采样模块） * 2
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         # 输出通道数不变，输出的signal_size变为signal_size*scale_factor（2）
                         nn.ReflectionPad1d(1),
                         nn.Conv1d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         # LayerNorm(int(ngf * mult / 2)),  # new 改成LN
                         ILN(int(ngf * mult / 2)),  # 使用的是ILN归一化
                         nn.LeakyReLU(0.2, True)]
            # i=0, 256->128（signal_size/4->signal_size/2）
            # i=1, 128->64(signal_size/2->signal_size)

        UpBlock2 += [nn.ReflectionPad1d(3),
                     nn.Conv1d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]
        # 64 -> output_nc

        self.DownBlock1 = nn.Sequential(*DownBlock1)
        self.DownBlock2 = nn.Sequential(*DownBlock2)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock1(input)
        x = self.DownBlock2(x)

        gap = torch.nn.functional.adaptive_avg_pool1d(x, 1)  # 自适应平均池化
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        #         gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gap = x * gap_weight.unsqueeze(2)

        gmp = torch.nn.functional.adaptive_max_pool1d(x, 1)  # 自适应最大池化
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        #         gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp = x * gmp_weight.unsqueeze(2)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)  # 特征图融合
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)  # 将所有通道数求和加一起，生成heatmap图

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool1d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
            # 将所有batch展开，输出为（batch_size,*）(adaILN使用MLP方法学习gamma, beta，参数学习依赖输入)
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


# In[9]:


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad1d(1),
                       nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm1d(dim),
                       nn.LeakyReLU(0.2, True)]

        conv_block += [nn.ReflectionPad1d(1),
                       nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm1d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad1d(1)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = LayerNorm(dim)
        # self.norm1 = adaILN(dim)  # adaILN归一化（可以改成GN或者LN试一试）
        self.relu1 = nn.LeakyReLU(0.2, True)

        self.pad2 = nn.ReflectionPad1d(1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        # self.norm2 = adaILN(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)  # -> [B, 256, 1024]
        out = self.norm1(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        # out = self.norm1(out, gamma, beta)
        ## IN & GN:out = self.norm1(out) LN:
        # out = self.norm1(out)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        # out = self.norm2(out, gamma, beta)
        # out = self.norm2(out)
        out = self.norm2(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        return out + x  # [B, 256, 1024]
"""
## LayerNorm:
def forward(self, x, gamma, beta):
    out = self.pad1(x)
    out = self.conv1(out)  # -> [B, 256, 1024]
    out = self.norm1(out.permute(0, 2, 1))
    out = out.permute(0, 2, 1)
    out = self.relu1(out)
    out = self.pad2(out)
    out = self.conv2(out)
    out = self.norm2(out.permute(0, 2, 1))
    out = out.permute(0, 2, 1)
    return out + x 
    
## IN or GN
    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)  # -> [B, 256, 1024]
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + x

## AdaLIN
out = self.norm1(out, gamma, beta)
"""

class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        # 创建一个可训练的Tensor
        self.rho = Parameter(torch.Tensor(1, num_features, 1))  # rho同样为可学习参数，作为IN与LN权重（IN*rho+LN*(1-rho)）
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=2, keepdim=True), torch.var(input, dim=2, keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2], keepdim=True), torch.var(input, dim=[1, 2], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1)) * out_ln
        #         out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        out = out * gamma.unsqueeze(2) + beta.unsqueeze(2)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        #         self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        #         self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        #         self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho = Parameter(torch.Tensor(1, num_features, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=2, keepdim=True), torch.var(input, dim=2, keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2], keepdim=True), torch.var(input, dim=[1, 2], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        #         out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
        #                 1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        #         out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        out = self.rho.expand(input.shape[0], -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1) + self.beta.expand(input.shape[0], -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad1d(1),
                 nn.utils.spectral_norm(
                     nn.Conv1d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]
        # input_nc->64,第1层下采样

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad1d(1),
                      nn.utils.spectral_norm(
                          nn.Conv1d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]
            # i=0,64->128;i=1,128->256;第2、3层下采样

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad1d(1),
                  # gan中常用的频谱归一化
                  nn.utils.spectral_norm(
                      nn.Conv1d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        # 256->512；第4层下采样(步长为1)

        # Class Activation Map（类别激活图CAM）
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        # 512->1024
        self.conv1x1 = nn.Conv1d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        # 1024->512
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad1d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv1d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool1d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2)

        gmp = torch.nn.functional.adaptive_max_pool1d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)  # 特征图融合
        x = self.leaky_relu(self.conv1x1(x))  # 将所有通道数求和加一起，生成heatmap图

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


# In[27]:


# # test
# Dnet = Discriminator(input_nc=1)
# a = Gout[0]
# print(a)
# Dout = Dnet(a)
# print(Dout)


# In[1]:


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
from glob import glob
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from MyDataset import *
from Net1 import *


class Test:
    def __init__(self, test_path='data/test', result_path='result', signal_size=4096, light=True,
                 input_nc=1, output_nc=1, ch=64, n_blocks=6, device='cuda:1', model_step=0):
        self.test_path = test_path
        self.result_path = result_path
        self.signal_size = signal_size
        self.light = light
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ch = ch
        self.n_blocks = n_blocks
        self.device = device
        self.model_step = model_step
        self.after_Encoder_Down_sampling = []
        self.after_Encoder_Bottleneck = []
        self.after_Decoder_UpBlock1_3 = []
        self.after_Decoder_UpBlock1_6 = []

    def dataload(self):
        Testdata = GetData(self.test_path)
        self.testA, self.testB = Testdata.get_data()
        self.testA_loader = DataLoader(GetDataset(self.testA), batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(GetDataset(self.testB), batch_size=1, shuffle=False)

    def build_model(self):
        self.genA2B = ResnetGenerator(input_nc=self.input_nc, output_nc=self.output_nc, ngf=self.ch,
                                      n_blocks=self.n_blocks, signal_size=self.signal_size, light=self.light).to(
            self.device)
        self.genB2A = ResnetGenerator(input_nc=self.input_nc, output_nc=self.output_nc, ngf=self.ch,
                                      n_blocks=self.n_blocks, signal_size=self.signal_size, light=self.light).to(
            self.device)
        self.disGA = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=5).to(self.device)

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, 'model_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def save_img(self, num, i):
        plt.subplot(1, 7, i)
        plt.plot(num)

    def save_resultsA(self, real_A, fake_A2A, fake_A2B, fake_A2B2A, fake_A2A_heatmap,
                      fake_A2B_heatmap, fake_A2B2A_heatmap, step, flag):
        path = os.path.join(os.path.join(self.result_path, 'test', 'result'), str(step), 'A-B', str(flag))
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.figure(figsize=(40, 3))

        np.savetxt(os.path.join(path, 'real_A' + str(step) + '.txt'), real_A[0][0].cpu().detach().numpy())
        self.save_img(real_A[0][0].cpu().detach().numpy(), 1)
        plt.title('real_A')

        np.savetxt(os.path.join(path, 'fake_A2A_heatmap' + str(step) + '.txt'),
                   fake_A2A_heatmap[0][0].cpu().detach().numpy())
        self.save_img(fake_A2A_heatmap[0][0].cpu().detach().numpy(), 2)
        plt.title('fake_A2A_heatmap')

        np.savetxt(os.path.join(path, 'fake_A2A' + str(step) + '.txt'), fake_A2A[0][0].cpu().detach().numpy())
        self.save_img(fake_A2A[0][0].cpu().detach().numpy(), 3)
        plt.title('fake_A2A')

        np.savetxt(os.path.join(path, 'fake_A2B_heatmap' + str(step) + '.txt'),
                   fake_A2B_heatmap[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B_heatmap[0][0].cpu().detach().numpy(), 4)
        plt.title('fake_A2B_heatmap')

        np.savetxt(os.path.join(path, 'fake_A2B' + str(step) + '.txt'), fake_A2B[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B[0][0].cpu().detach().numpy(), 5)
        plt.title('fake_A2B')

        np.savetxt(os.path.join(path, 'fake_A2B2A_heatmap' + str(step) + '.txt'),
                   fake_A2B2A_heatmap[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B2A_heatmap[0][0].cpu().detach().numpy(), 6)
        plt.title('fake_A2B2A_heatmap')

        np.savetxt(os.path.join(path, 'fake_A2B2A' + str(step) + '.txt'), fake_A2B2A[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B2A[0][0].cpu().detach().numpy(), 7)
        plt.title('fake_A2B2A')

        plt.savefig(os.path.join(path, str(step) + '.png'), dpi=600)
        plt.cla()

    def save_resultsB(self, real_B, fake_B2B, fake_B2A, fake_B2A2B, fake_B2B_heatmap,
                      fake_B2A_heatmap, fake_B2A2B_heatmap, step, flag):
        path = os.path.join(os.path.join(self.result_path, 'test', 'result'), str(step), 'B-A', str(flag))
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.figure(figsize=(40, 3))

        np.savetxt(os.path.join(path, 'real_B' + str(step) + '.txt'), real_B[0][0].cpu().detach().numpy())
        self.save_img(real_B[0][0].cpu().detach().numpy(), 1)
        plt.title('real_B')

        np.savetxt(os.path.join(path, 'fake_B2B_heatmap' + str(step) + '.txt'),
                   fake_B2B_heatmap[0][0].detach().cpu().numpy())
        self.save_img(fake_B2B_heatmap[0][0].detach().cpu().numpy(), 2)
        plt.title('fake_B2B_heatmap')

        np.savetxt(os.path.join(path, 'fake_B2B' + str(step) + '.txt'), fake_B2B[0][0].cpu().detach().numpy())
        self.save_img(fake_B2B[0][0].cpu().detach().numpy(), 3)
        plt.title('fake_B2B')

        np.savetxt(os.path.join(path, 'fake_B2A_heatmap' + str(step) + '.txt'),
                   fake_B2A_heatmap[0][0].detach().cpu().numpy())
        self.save_img(fake_B2A_heatmap[0][0].detach().cpu().numpy(), 4)
        plt.title('fake_B2A_heatmap')

        np.savetxt(os.path.join(path, 'fake_B2A' + str(step) + '.txt'), fake_B2A[0][0].cpu().detach().numpy())
        self.save_img(fake_B2A[0][0].cpu().detach().numpy(), 5)
        plt.title('fake_B2A')

        np.savetxt(os.path.join(path, 'fake_B2A2B_heatmap' + str(step) + '.txt'),
                   fake_B2A2B_heatmap[0][0].detach().cpu().numpy())
        self.save_img(fake_B2A2B_heatmap[0][0].detach().cpu().numpy(), 6)
        plt.title('fake_B2A2B_heatmap')

        np.savetxt(os.path.join(path, 'fake_B2A2B' + str(step) + '.txt'), fake_B2A2B[0][0].cpu().detach().numpy())
        self.save_img(fake_B2A2B[0][0].cpu().detach().numpy(), 7)
        plt.title('fake_B2A2B')

        plt.savefig(os.path.join(path, str(step) + '.png'), dpi=600)
        plt.cla()

    def hook1(self, module, input, output):
        self.after_Encoder_Down_sampling.append(output.clone().detach().cpu().numpy())

    def hook2(self, module, input, output):
        self.after_Encoder_Bottleneck.append(output.clone().detach().cpu().numpy())

    def hook3(self, module, input, output):
        self.after_Decoder_UpBlock1_3.append(output.clone().detach().cpu().numpy())

    def hook4(self, module, input, output):
        self.after_Decoder_UpBlock1_6.append(output.clone().detach().cpu().numpy())

    def save_heatmap_img(self, num, i):
        plt.subplot(1, 6, i)
        plt.plot(num)

    def save_heatmap_resultsA(self, real_A, fake_A2B, step, flag):
        path = os.path.join(os.path.join(self.result_path, 'test', 'heatmap'), str(step), 'A-B', str(flag))
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.figure(figsize=(36, 3))
        np.savetxt(os.path.join(path, 'real_A' + str(step) + '.txt'), real_A[0][0].cpu().detach().numpy())
        self.save_heatmap_img(real_A[0][0].cpu().detach().numpy(), 1)
        plt.title('real_A')

        np.savetxt(os.path.join(path, 'after_Encoder_Down_sampling' + str(step) + '.txt'),
                   self.after_Encoder_Down_sampling[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Encoder_Down_sampling[flag - 1].sum(axis=1)[0], 2)
        plt.title('after_Encoder_Down_samplin')

        np.savetxt(os.path.join(path, 'after_Encoder_Bottleneck' + str(step) + '.txt'),
                   self.after_Encoder_Bottleneck[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Encoder_Bottleneck[flag - 1].sum(axis=1)[0], 3)
        plt.title('after_Encoder_Bottleneck')

        np.savetxt(os.path.join(path, 'after_Decoder_UpBlock1_3' + str(step) + '.txt'),
                   self.after_Decoder_UpBlock1_3[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Decoder_UpBlock1_3[flag - 1].sum(axis=1)[0], 4)
        plt.title('after_Decoder_UpBlock1_3')

        np.savetxt(os.path.join(path, 'after_Decoder_UpBlock1_6' + str(step) + '.txt'),
                   self.after_Decoder_UpBlock1_6[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Decoder_UpBlock1_6[flag - 1].sum(axis=1)[0], 5)
        plt.title('after_Decoder_UpBlock1_6')

        np.savetxt(os.path.join(path, 'fake_A2B' + str(step) + '.txt'), fake_A2B[0][0].cpu().detach().numpy())
        self.save_heatmap_img(fake_A2B[0][0].cpu().detach().numpy(), 6)
        plt.title('fake_A2B')

        plt.savefig(os.path.join(path, str(step) + '.png'), dpi=600)
        plt.cla()

    def save_heatmap_resultsB(self, real_B, fake_B2A, step, flag):
        path = os.path.join(os.path.join(self.result_path, 'test', 'heatmap'), str(step), 'B-A', str(flag))
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.figure(figsize=(36, 3))

        np.savetxt(os.path.join(path, 'real_B' + str(step) + '.txt'), real_B[0][0].cpu().detach().numpy())
        self.save_heatmap_img(real_B[0][0].cpu().detach().numpy(), 1)
        plt.title('real_B')

        np.savetxt(os.path.join(path, 'after_Encoder_Down_sampling' + str(step) + '.txt'),
                   self.after_Encoder_Down_sampling[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Encoder_Down_sampling[flag - 1].sum(axis=1)[0], 2)
        plt.title('after_Encoder_Down_samplin')

        np.savetxt(os.path.join(path, 'after_Encoder_Bottleneck' + str(step) + '.txt'),
                   self.after_Encoder_Bottleneck[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Encoder_Bottleneck[flag - 1].sum(axis=1)[0], 3)
        plt.title('after_Encoder_Bottleneck')

        np.savetxt(os.path.join(path, 'after_Decoder_UpBlock1_3' + str(step) + '.txt'),
                   self.after_Decoder_UpBlock1_3[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Decoder_UpBlock1_3[flag - 1].sum(axis=1)[0], 4)
        plt.title('after_Decoder_UpBlock1_3')

        np.savetxt(os.path.join(path, 'after_Decoder_UpBlock1_6' + str(step) + '.txt'),
                   self.after_Decoder_UpBlock1_6[flag - 1].sum(axis=1)[0])
        self.save_heatmap_img(self.after_Decoder_UpBlock1_6[flag - 1].sum(axis=1)[0], 5)
        plt.title('after_Decoder_UpBlock1_6')

        np.savetxt(os.path.join(path, 'fake_B2A' + str(step) + '.txt'), fake_B2A[0][0].cpu().detach().numpy())
        self.save_heatmap_img(fake_B2A[0][0].cpu().detach().numpy(), 6)
        plt.title('fake_B2A')

        plt.savefig(os.path.join(path, str(step) + '.png'), dpi=600)
        plt.cla()

    def test(self):
        model_list = glob(os.path.join(self.result_path, 'models', '*.pt'))
        model_step = 0
        if not len(model_list) == 0:
            if self.model_step and os.path.exists(os.path.join(os.path.join(self.result_path, 'models'),
                                                               'model_params_%07d.pt' % self.model_step)):
                model_step = self.model_step
            else:
                model_list.sort()
                model_step = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load_model(os.path.join(self.result_path, 'models'), model_step)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        #         print(self.genA2B.state_dict)
        handle1 = self.genA2B.DownBlock1.register_forward_hook(self.hook1)
        handle2 = self.genA2B.DownBlock2.register_forward_hook(self.hook2)
        handle3 = self.genA2B.UpBlock1_3.register_forward_hook(self.hook3)
        handle4 = self.genA2B.UpBlock1_6.register_forward_hook(self.hook4)
        self.after_Encoder_Down_sampling = []
        self.after_Encoder_Bottleneck = []
        self.after_Decoder_UpBlock1_3 = []
        self.after_Decoder_UpBlock1_6 = []
        for i, real_A in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)
            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
            self.save_resultsA(real_A, fake_A2A, fake_A2B, fake_A2B2A, fake_A2A_heatmap,
                               fake_A2B_heatmap, fake_A2B2A_heatmap, model_step, i + 1)
            self.save_heatmap_resultsA(real_A, fake_A2B, model_step, i + 1)
            if i >= 50:
                break

        self.after_Encoder_Down_sampling = []
        self.after_Encoder_Bottleneck = []
        self.after_Decoder_UpBlock1_3 = []
        self.after_Decoder_UpBlock1_6 = []
        for i, real_B in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
            self.save_resultsB(real_B, fake_B2B, fake_B2A, fake_B2A2B, fake_B2B_heatmap,
                               fake_B2A_heatmap, fake_B2A2B_heatmap, model_step, i + 1)
            self.save_heatmap_resultsB(real_B, fake_B2A, model_step, i + 1)
            if i >= 50:
                break

        handle1.remove()
        handle2.remove()
        handle3.remove()
        handle4.remove()

if __name__ == '__main__':
    gan = Test(test_path='data/test', result_path='result/net1', signal_size=4096, light=True,
               input_nc=1, output_nc=1, ch=64, n_blocks=6, device='cuda:2', model_step=50000)
    gan.dataload()
    gan.build_model()
    gan.test()
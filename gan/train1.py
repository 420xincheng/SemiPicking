#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import time
import itertools
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from glob import glob
from MyDataset import *
from Net1 import *


# In[2]:


class Train:
    def __init__(self, train_path='data/train', result_path='result', signal_size=4096, num_epochs=5000000,
                 batch_size=4, light=True,
                 input_nc=1, output_nc=1, ch=64, n_blocks=6, lr=1e-4, weight_decay=1e-4, adv_weight=1, cycle_weight=10,
                 identity_weight=10, cam_weight=1000, decay_flag=True, device='cuda:0', resume=False):
        self.train_path = train_path
        self.result_path = result_path
        self.signal_size = signal_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.light = light
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ch = ch
        self.n_blocks = n_blocks
        self.lr = lr
        self.weight_decay = weight_decay
        self.adv_weight = adv_weight
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight
        self.cam_weight = cam_weight
        self.decay_flag = decay_flag
        self.device = device
        self.resume = resume
        os.makedirs(result_path, exist_ok=True)

    def dataload(self):
        Traindata = GetData(self.train_path)
        self.trainA, self.trainB = Traindata.get_data()
        #         self.scaler = Traindata.get_scaler()
        self.trainA_loader = DataLoader(GetDataset(self.trainA), batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(GetDataset(self.trainB), batch_size=self.batch_size, shuffle=True)

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

    def define_loss(self):
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

    def define_optim(self):
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                        lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                                                        self.disLA.parameters(), self.disLB.parameters()),
                                        lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))

    def define_rho(self):
        self.Rho_clipper = RhoClipper(0, 1)

    def save_model(self, path, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(path, 'model_params_%07d.pt' % step))

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, 'model_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def save_img(self, num, i):
        # plt.plot(7, 2, i)
        plt.subplot(2, 7, i)
        plt.plot(num)

    def save_intermediate_resultsA(self, real_A, fake_A2A, fake_A2B, fake_A2B2A, fake_A2A_heatmap,
                                   fake_A2B_heatmap, fake_A2B2A_heatmap, step, flag, A_label):
        path = os.path.join(os.path.join(self.result_path, 'train'), str(step), str(flag))
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.cla()
        plt.figure(figsize=(40, 5))
        np.savetxt(os.path.join(path, 'real_A' + str(step) + '.txt'), real_A[0][0].cpu().detach().numpy())
        self.save_img(real_A[0][0].cpu().detach().numpy(), 1)

        # np.savetxt(os.path.join(path, 'fake_A2A_heatmap' + str(step) + '.txt'),
        #            fake_A2A_heatmap[0][0].cpu().detach().numpy())
        self.save_img(fake_A2A_heatmap[0][0].cpu().detach().numpy(), 2)

        # np.savetxt(os.path.join(path, 'fake_A2A' + str(step) + '.txt'), fake_A2A[0][0].cpu().detach().numpy())
        self.save_img(fake_A2A[0][0].cpu().detach().numpy(), 3)

        # np.savetxt(os.path.join(path, 'fake_A2B_heatmap' + str(step) + '.txt'),
        #            fake_A2B_heatmap[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B_heatmap[0][0].cpu().detach().numpy(), 4)

        # np.savetxt(os.path.join(path, 'fake_A2B' + str(step) + '.txt'), fake_A2B[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B[0][0].cpu().detach().numpy(), 5)

        # np.savetxt(os.path.join(path, 'fake_A2B2A_heatmap' + str(step) + '.txt'),
        #            fake_A2B2A_heatmap[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B2A_heatmap[0][0].cpu().detach().numpy(), 6)

        np.savetxt(os.path.join(path, 'fake_A2B2A' + str(step) + '.txt'), fake_A2B2A[0][0].cpu().detach().numpy())
        self.save_img(fake_A2B2A[0][0].cpu().detach().numpy(), 7)

        real_l = A_label[0].cpu().detach().numpy()
        real_l = [real_l]
        np.savetxt(os.path.join(path, 'real_label_' + str(real_l) + str(step) + '.txt'), real_l)

    def save_intermediate_resultsB(self, real_B, fake_B2B, fake_B2A, fake_B2A2B, fake_B2B_heatmap,
                                   fake_B2A_heatmap, fake_B2A2B_heatmap, step, flag):
        path = os.path.join(os.path.join(self.result_path, 'train'), str(step), str(flag))
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        np.savetxt(os.path.join(path, 'real_B' + str(step) + '.txt'), real_B[0][0].cpu().detach().numpy())
        self.save_img(real_B[0][0].cpu().detach().numpy(), 8)

        # np.savetxt(os.path.join(path, 'fake_B2B_heatmap' + str(step) + '.txt'),
        #            fake_B2B_heatmap[0][0].detach().cpu().numpy())
        self.save_img(fake_B2B_heatmap[0][0].detach().cpu().numpy(), 9)

        # np.savetxt(os.path.join(path, 'fake_B2B' + str(step) + '.txt'), fake_B2B[0][0].cpu().detach().numpy())
        self.save_img(fake_B2B[0][0].cpu().detach().numpy(), 10)

        # np.savetxt(os.path.join(path, 'fake_B2A_heatmap' + str(step) + '.txt'),
        #            fake_B2A_heatmap[0][0].detach().cpu().numpy())
        self.save_img(fake_B2A_heatmap[0][0].detach().cpu().numpy(), 11)

        # np.savetxt(os.path.join(path, 'fake_B2A' + str(step) + '.txt'), fake_B2A[0][0].cpu().detach().numpy())
        self.save_img(fake_B2A[0][0].cpu().detach().numpy(), 12)

        # np.savetxt(os.path.join(path, 'fake_B2A2B_heatmap' + str(step) + '.txt'),
        #            fake_B2A2B_heatmap[0][0].detach().cpu().numpy())
        self.save_img(fake_B2A2B_heatmap[0][0].detach().cpu().numpy(), 13)

        # np.savetxt(os.path.join(path, 'fake_B2A2B' + str(step) + '.txt'), fake_B2A2B[0][0].cpu().detach().numpy())
        self.save_img(fake_B2A2B[0][0].cpu().detach().numpy(), 14)

        plt.savefig(os.path.join(path, str(step) + '.png'), dpi=600)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_step = 1
        # 续训
        if self.resume:
            model_list = glob(os.path.join(self.result_path, 'models', '*.pt'))
            if len(model_list) != 0:
                model_list.sort()
                start_step = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load_model(os.path.join(self.result_path, 'models'), start_step)
                print("load success!")
                # 学习率衰减
                if self.decay_flag and start_step > (self.num_epochs // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.num_epochs // 2)) * (
                            start_step - self.num_epochs // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.num_epochs // 2)) * (
                            start_step - self.num_epochs // 2)

        print("training start!")
        start_time = time.time()
        # 学习率衰减
        for step in range(start_step, self.num_epochs + 1):
            if self.decay_flag and step > (self.num_epochs // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.num_epochs // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.num_epochs // 2))

            try:
                real_A_dict = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A_dict = trainA_iter.next()

            try:
                real_B = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B = trainB_iter.next()

            real_A, real_B = real_A_dict['signal'].to(self.device), real_B.to(self.device)

            # D_loss
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(
                fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit,
                                             torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(
                fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(
                fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit,
                                             torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(
                fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(
                fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit,
                                             torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(
                fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(
                fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit,
                                             torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(
                fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
            Discriminator_loss = D_loss_A + D_loss_B

            Discriminator_loss.backward()
            self.D_optim.step()

            # G_loss
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit,
                                         torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(
                fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit,
                                         torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(
                fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A = self.adv_weight * (
                    G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (
                    G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B

            with open(os.path.join(self.result_path, 'G_adv_loss.txt'), 'a') as f:
                f.write(str((
                                    (G_ad_loss_GA + G_ad_loss_LA) + (G_ad_loss_GB + G_ad_loss_LB)
                            ).item()) + '\n')

            with open(os.path.join(self.result_path, 'D_adv_loss.txt'), 'a') as f:
                f.write(str((
                                    (D_ad_loss_GA + D_ad_loss_LA) + (D_ad_loss_GB + D_ad_loss_LB)
                            ).item()) + '\n')

            with open(os.path.join(self.result_path, 'G_cam_loss.txt'), 'a') as f:
                f.write(str((
                                    (G_ad_cam_loss_GA + G_ad_cam_loss_LA) + (G_ad_cam_loss_GB + G_ad_cam_loss_LB)
                            ).item()) + '\n')

            with open(os.path.join(self.result_path, 'D_cam_loss.txt'), 'a') as f:
                f.write(str((
                                    (D_ad_cam_loss_GA + D_ad_cam_loss_LA) + (D_ad_cam_loss_GB + D_ad_cam_loss_LB)
                            ).item()) + '\n')

            with open(os.path.join(self.result_path, 'G_cycle_loss.txt'), 'a') as f:
                f.write(str((
                                    G_recon_loss_A + G_recon_loss_B
                            ).item()) + '\n')

            with open(os.path.join(self.result_path, 'G_identity_loss.txt'), 'a') as f:
                f.write(str((
                                    G_identity_loss_A + G_identity_loss_B
                            ).item()) + '\n')

            Generator_loss.backward()
            self.G_optim.step()

            # 更新adaILN的rho参数
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)
            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                step, self.num_epochs, time.time() - start_time, Discriminator_loss, Generator_loss))

            # 测试
            if step % 1000 == 0 and step >= 10000:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.signal_size * 7, 0, 1))
                B2A = np.zeros((self.signal_size * 7, 0, 1))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for i in range(train_sample_num):
                    try:
                        real_A_dict = trainA_iter.next()
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A_dict = trainA_iter.next()

                    try:
                        real_B = trainB_iter.next()
                    except:
                        trainB_iter = iter(self.trainB_loader)
                        real_B = trainB_iter.next()
                    real_A, real_B = real_A_dict['signal'].to(self.device), real_B.to(self.device)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    self.save_intermediate_resultsA(real_A, fake_A2A, fake_A2B, fake_A2B2A, fake_A2A_heatmap,
                                                    fake_A2B_heatmap, fake_A2B2A_heatmap, step, i + 1, real_A_dict['label'])
                    self.save_intermediate_resultsB(real_B, fake_B2B, fake_B2A, fake_B2A2B, fake_B2B_heatmap,
                                                    fake_B2A_heatmap, fake_B2A2B_heatmap, step, i + 1)

                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % 10000 == 0:
                self.save_model(os.path.join(self.result_path, 'models'), step)
            if step % 1000 == 0:
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                params['disLA'] = self.disLA.state_dict()
                params['disLB'] = self.disLB.state_dict()
                torch.save(params, os.path.join(self.result_path, 'model_params_%07d.pt' % step))


if __name__ == '__main__':
    gan = Train(train_path='data/train', result_path='result/new', signal_size=4096, num_epochs=20000, batch_size=4,
                light=True,
                input_nc=1, output_nc=1, ch=64, n_blocks=6, lr=1e-4, weight_decay=1e-4, adv_weight=1, cycle_weight=10,
                identity_weight=10, cam_weight=1000, decay_flag=True, device='cuda:3', resume=True)
    gan.dataload()
    gan.build_model()
    gan.define_loss()
    gan.define_optim()
    gan.define_rho()
    gan.train()
    print("training finished!")
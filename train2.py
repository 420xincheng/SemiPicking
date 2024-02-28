import logging
import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from my_dataset import MyDataset, RandomGenerator
from models.net import TransUnet, CONFIGS
from datetime import datetime

from utils.utils import *

DATA_ROOT_PATH = './dataset/'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def get_acc(out, label):
    num_correct = 0
    _, pred_label = out.max(1)
    for i in range(len(label)):
        if (
                np.abs(
                    pred_label[i].cpu().detach().numpy()
                    - label[i].cpu().detach().numpy()
                )
                <= 5
        ):
            num_correct += 1
    return num_correct


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class Train:
    def __init__(self, train='train', seed=666, batch_size=4, n_gpu=1, base_lr=0.001, max_epoch=10,
                 weight_decay=0, warmup=20, construct='BASELINE', l_sspcab=0.1,
                 output_size=4096, output_path='./output', add_valid=False, add_sspcab=True):
        self.mode = train
        self.train_path = DATA_ROOT_PATH + train  # ./dataset/train
        self.seed = seed
        self.batch_size = batch_size
        self.n_gpu = n_gpu
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.output_size = output_size
        self.output_path = output_path
        self.add_valid = add_valid  # 是否启用验证集
        self.add_sspcab = add_sspcab
        self.warmup = warmup
        self.construct = construct
        self.l_sspcab = l_sspcab

    def save_outputs(self, output, sample_id, true_label, hidden_map):
        os.makedirs(self.output_path, exist_ok=True)
        array_path = os.path.join(self.output_path, f'softmax/{TIMESTAMP}')  # 输出
        map_path = os.path.join(self.output_path, f'feature_map/{TIMESTAMP}')  # hidden_map输出
        os.makedirs(array_path, exist_ok=True)
        os.makedirs(map_path, exist_ok=True)
        np.savetxt(os.path.join(array_path, f'{sample_id}_{true_label}.txt'), output.cpu().detach().numpy())
        for i in range(min(hidden_map.size()[0], self.batch_size)):
            # hidden_map = hidden_map.squeeze()  # [1,512.256] -> [512.256]
            np.savetxt(os.path.join(map_path, f'{sample_id}_{true_label}.txt'),
                       hidden_map[i, :, :].cpu().detach().numpy())

    def train(self, trans_u_net):
        sm_path = DATA_ROOT_PATH + f'log/{self.mode}_log/{TIMESTAMP}'
        os.makedirs(sm_path, exist_ok=True)
        logging.basicConfig(filename=sm_path + f'/log_{self.mode}.txt', level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        writer = SummaryWriter(sm_path)
        getDataPath = './dataset'
        db_train = MyDataset(data_dir=getDataPath, data_source='train', corrupt_per=0, cor_num=100,
                             transform=transforms.Compose([RandomGenerator(output_size=self.output_size)]))
        train_loader = DataLoader(db_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        print("The length of train set {} is: {}".format(getDataPath, len(db_train)))
        lb_smooth = 0.1
        # 验证集
        if self.add_valid:
            db_valid = MyDataset(data_dir=getDataPath, data_source='valid',
                                 transform=transforms.Compose([RandomGenerator(output_size=self.output_size)]))
            valid_loader = DataLoader(db_valid, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if self.n_gpu > 1:
            print(f"train with {self.n_gpu}")
            trans_u_net = nn.DataParallel(trans_u_net)
        # trans_u_net.train()
        max_epoch = self.max_epoch
        max_iterations = max_epoch * len(train_loader)  # len(train_loader) = len(train)/batch_size
        base_lr = self.base_lr

        cross_loss = LabelSmoothingCrossEntropy(lb_smooth)
        # cross_loss = CrossEntropyLoss()
        # optimizer = optim.Adagrad(trans_u_net.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        optimizer = optim.Adam(trans_u_net.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        # optimizer = optim.SGD(trans_u_net.parameters(), lr=self.base_lr, weight_decay=5E-5)
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lrf = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / max_epoch)) / 2) * (1 - self.lrf) + self.lrf  # cosine
        # scheduler = LambdaLR(optimizer, lr_lambda=lf)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup, max_epoch)
        logging.info(
            f'{TIMESTAMP} Res contain sspcab optim = {optimizer} , trans layer=4, label_smoothing={lb_smooth}  \nmodel = {self.construct}, add valid = {self.add_valid}, add sspcab = {self.add_sspcab}, batch = {self.batch_size}, weight_decay = {self.weight_decay},lr = {self.base_lr}, seed = {self.seed}, len valid {len(valid_loader)}'
        )
        logging.info("{} iterations per epoch. {} max iterations. train loader length {} \n".format(len(train_loader),
                                                                                                    max_iterations,
                                                                                                    train_loader.__len__()))
        iterator = tqdm(range(max_epoch), ncols=max_epoch)
        iter_num, iter_val = 0, 0
        best_acc = 0.
        out_train_acc, out_train_loss, out_valid_acc, out_valid_loss = [], [], [], []
        start_epoch = 0
        for epoch_num in range(start_epoch, max_epoch):
            train_total = 0
            train_correct = 0
            train_loss = 0.
            trans_u_net.train()
            for id_batch, data in enumerate(train_loader):  # 所有数据集进行训练
                signals, labels = data['signal'], data['label']
                signals, labels = signals.to(device), labels.to(device)
                signals = signals.long()  

                outputs, hidden_map, cost_sspcab = trans_u_net(
                    signals)  # out_size [1,4096]  

                loss_ce = cross_loss(outputs, labels)  # out_size = [B,L] labels.size() = [1]  # [B,4096]

                loss = loss_ce + self.l_sspcab * cost_sspcab
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learning_rate = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param in optimizer.param_groups:
                    param['lr'] = learning_rate
                _, predicted = outputs.max(1)
                # train_correct += (predicted == labels).sum().item()

                train_correct += get_acc(outputs, labels)
                train_loss += loss.item()
                train_total += labels.size(0)  # 获取batch大小

                iter_num += 1

                # 只保存最后一个样本的输出-进行可视化
                # if iter_num == max_iterations:
                #     self.save_outputs(output=outputs, sample_id=data['case_name'], true_label=labels,
                #                       hidden_map=hidden_map)

                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                writer.add_scalar('iteration/lr', learning_rate, iter_num)
                # writer.add_scalar('iteration/lr', optimizer.param_groups[0]["lr"], iter_num)
                writer.add_scalar('iteration/loss', loss, iter_num)
                writer.add_scalar('iteration/sspcab_loss', cost_sspcab, iter_num)

            # 每跑完一次epoch测试一下准确率
            train_accuracy = train_correct / train_total
            train_loss = train_loss / len(train_loader)
            logging.info('epoch %d , train_acc : %f%%, train_loss : %.6f' % (
                epoch_num, train_accuracy * 100, train_loss))
            out_train_acc.append(train_accuracy)
            out_train_loss.append(train_loss)
            writer.add_scalar('epoch/train_acc', train_accuracy, epoch_num)
            writer.add_scalar('epoch/train_loss', train_loss, epoch_num)

            # valid
            if self.add_valid:
                trans_u_net.eval()
                val_acc = torch.zeros(1).to(device)  # 累计预测正确的样本数
                val_loss = torch.zeros(1).to(device)
                valid_num = 0
                with torch.no_grad():
                    for step, data in enumerate(valid_loader):
                        signal, labels = data['signal'], data['label']
                        signal, labels = signal.to(device), labels.to(device)
                        valid_num += signal.size(0)  # 获取批大小
                        pred, hidden_map, cost_sspcab_val = trans_u_net(signal)
                        first_arrival_point = torch.argmax(torch.softmax(pred, -1), dim=-1)  # 验证集概率为0
                        val_acc += get_acc(pred, labels)
                        loss = cross_loss(pred, labels)
                        writer.add_scalar('iteration/valid_loss', loss, iter_val)
                        writer.add_scalar('iteration/valid_ss_loss', cost_sspcab_val, iter_val)
                        iter_val += 1
                        loss += self.l_sspcab * cost_sspcab_val
                        val_loss += loss
                    val_acc = val_acc.item() / valid_num
                    val_loss = val_loss.item() / (step + 1)
                    writer.add_scalar('epoch/valid_acc', val_acc, epoch_num)
                    writer.add_scalar('epoch/valid_loss', val_loss, epoch_num)
                    out_valid_acc.append(val_acc)
                    out_valid_loss.append(val_loss)
                    logging.info('epoch %d, valid_acc : %f%%, valid_loss : %.6f' % (
                        epoch_num, val_acc * 100, val_loss))

            save_interval = 50  # 保存周期
            weight_save_path = f'./weights/{TIMESTAMP}/'
            os.makedirs(weight_save_path, exist_ok=True)

            if best_acc < train_accuracy:  # 保存最优模型
                best_acc = train_accuracy
                torch.save(trans_u_net.state_dict(), weight_save_path + f"best_model.pth")
            if epoch_num >= max_epoch - 1:  # 最后一期保存
                iterator.close()
                break
        writer.close()
        np.savetxt(sm_path + f'/improved_train_acc.txt', out_train_acc)
        np.savetxt(sm_path + f'/improved_train_loss.txt', out_train_loss)
        np.savetxt(sm_path + f'/improved_valid_acc.txt', out_valid_acc)
        np.savetxt(sm_path + f'/improved_valid_loss.txt', out_valid_loss)

        return "Training has finished!"


if __name__ == '__main__':
    SEED = 2023 # change
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    con = 'Res50_b16'  # 加sspcab
    model = TransUnet(CONFIGS[con], signal_size=4096, num_classes=1, add_sspcab=True).to(device)

    trainer = Train(train='train', seed=SEED, batch_size=32, max_epoch=200, base_lr=1e-4, l_sspcab=0.1,
                    weight_decay=5e-3, add_valid=True, construct=con, add_sspcab=True)
    trainer.train(model)
    # 0和0.01的lr=-3的情况下

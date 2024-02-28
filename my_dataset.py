import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import maxabs_scale


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 4096
        self.ones = torch.eye(output_size)  # 标签平滑处理变为onehot

    def __call__(self, sample):
        signal, label = sample['signal'], sample['label']
        # signal = maxabs_scale(signal)
        x = signal.size  # 4096
        if x != self.output_size:
            signal = zoom(signal, (self.output_size / x), order=3)
        signal = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0) 
        label = torch.from_numpy(label.astype(np.int64)) - 1
        sample = {'signal': signal, 'label': label.long()}
        return sample


class MyDataset(Dataset):
    def __init__(self, data_dir, data_source, transform=None, ici_label_dict=None, corrupt_per=0, cor_num=100):
        self.transform = transform
        self.data_dir = data_dir  # "./dataset"
        self.data_source = data_source  # 'train' or 'valid' or 'test'
        self.list_path = osp.join(data_dir, data_source)  # './dataset/train/'
        self.sample_list = open(osp.join(data_dir, self.data_source + '.txt')).readlines()
        self.ici_label_dict = ici_label_dict
        self.xtrain, self.xlabel = [], []
        for i in range(len(self.sample_list)):
            fname = self.sample_list[i].strip('\n')
            data_path = osp.join(self.list_path, fname)
            if self.data_source in ['valid'] or self.data_dir == './ici/data/dataset':
                data_path = f"./dataset/train/{fname}"
            data = np.load(data_path)
            signal, label = data['signal'], data['label']
            self.xtrain.append(signal)
            self.xlabel.append(label)
        if corrupt_per > 0:
            corrupt_labels = int(cor_num * corrupt_per)
            corrupt_labels = min(corrupt_labels, cor_num)
            print("Corrupting %d labels." % corrupt_labels)
            for i in range(corrupt_labels):
                self.xlabel[i] = np.array(np.random.randint(0, 4096))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        signal, label = self.xtrain[idx], self.xlabel[idx]
        sample_name = self.sample_list[idx].strip('\n')  # 样本名
        if self.ici_label_dict is not None:
            p_label = self.ici_label_dict.get(sample_name.split('.')[0])
            label = np.array(p_label)
        sample = {'signal': signal, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = sample_name
        return sample


DATA_DIR = '/www/signal/simtxt'
DATA_ROOT = '/www/signal/dataset'


class SignalNet_V1(Dataset):
    def __init__(self, mode='train', root=DATA_DIR, label=True, return_index=False, corrupt_per=0, cor_num=100, **kwargs):
        super(SignalNet, self).__init__()
        self.dataset_dir = root
        self.return_index = return_index
        self.unlabel = not label
        self.mode = mode
        if label:
            self.train_dir = osp.join(self.dataset_dir, 'train.txt')  # ./ici/data/dataset/train.txt
        else:
            self.train_dir = osp.join(self.dataset_dir, 'train_u.txt')
        self.val_dir = osp.join(self.dataset_dir, 'val.txt')
        self.test_dir = osp.join(self.dataset_dir, 'test.txt')
        if mode == 'train':
            self.sample_list = open(self.train_dir).readlines()
        elif mode == 'test':
            self.sample_list = open(self.test_dir).readlines()
        elif mode == 'val':
            self.sample_list = open(self.val_dir).readlines()
        self.xtrain, self.xlabel = [], []
        for i in range(len(self.sample_list)):
            fname = self.sample_list[i].strip('\n')
            filename = osp.join(DATA_DIR1 + '/train', fname)
            data = np.load(filename)  # './dataset/train/1.npz'
            signal, label = data['signal'], data['label']  # signal:[4096] label:1
            self.xtrain.append(signal)
            self.xlabel.append(label)
        if corrupt_per > 0:
            corrupt_labels = int(cor_num * corrupt_per)
            corrupt_labels = min(corrupt_labels, cor_num)
            print("Corrupting %d labels." % corrupt_labels)
            for i in range(corrupt_labels):
                self.xlabel[i] = np.random.randint(0, 4096)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        fname = self.sample_list[idx].strip('\n')
        signal, label = self.xtrain[idx], self.xlabel[idx]
        signal = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)
        if type(label) is int:
            label = torch.tensor(label - 1)
        else:
            label = torch.from_numpy(label.astype(np.int64)) - 1
        if self.return_index:
            return signal, label, idx
        if self.unlabel:  # 无标签
            ns = int(fname.split('.')[0])
            fname = str(ns) + '.npz'
            d = np.load(DATA_DIR1 + f'/gan/n15_{fname}')
            signal_noisy = d['signal']
            signal_noisy = torch.from_numpy(signal_noisy.astype(np.float32)).unsqueeze(0)
            w_n_signal = signal, signal_noisy
            return w_n_signal, label.long()
        return signal, label.long()


class SignalNet(Dataset):
    def __init__(self, mode='train', root=DATA_DIR, label=True, return_index=False, corrupt_per=0, cor_num=100, **kwargs):
        super(SignalNet, self).__init__()
        self.dataset_dir = root
        self.return_index = return_index
        self.unlabel = not label
        self.mode = mode
        if label:
            self.train_dir = osp.join(self.dataset_dir, 'train.txt')  
        else:
            self.train_dir = osp.join(self.dataset_dir, 'train_u.txt')
        self.val_dir = osp.join(self.dataset_dir, 'val.txt')
        self.test_dir = osp.join(self.dataset_dir, 'test.txt')
        if mode == 'train':
            self.sample_list = open(self.train_dir).readlines()
        elif mode == 'test':
            self.sample_list = open(self.test_dir).readlines()
        elif mode == 'val':
            self.sample_list = open(self.val_dir).readlines()

        # 为了方便打乱标签
        self.xtrain, self.xlabel = [], []
        for filename in self.sample_list:
            filename = filename.strip('\n')
            data = np.load(filename)  # './dataset/train/1.npz'
            signal, label = data['signal'], data['label']  # signal:[4096] label:1
            self.xtrain.append(signal)
            self.xlabel.append(label)

        if corrupt_per > 0:
            corrupt_labels = int(cor_num * corrupt_per)
            corrupt_labels = min(corrupt_labels, cor_num)
            print("Corrupting %d labels." % corrupt_labels)
            for i in range(corrupt_labels):
                self.xlabel[i] = np.random.randint(0, 4096)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        signal, label = self.xtrain[idx], self.xlabel[idx]
        signal = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)
        if type(label) is int:
            label = torch.tensor(label - 1)
        else:
            label = torch.from_numpy(label.astype(np.int64)) - 1

        if self.unlabel:  # 无标签, 返回signal, strong signal, label
            nidx = np.random.randint(1, 1200)
            data = np.load(f'{DATA_ROOT}/gan/n15_{nidx}.npz')
            signal_noisy = data['signal']
            signal_noisy = torch.from_numpy(signal_noisy.astype(np.float32)).unsqueeze(0)
            w_n_signal = signal, signal_noisy
            return w_n_signal, label.long()
        
        if self.return_index:
            return signal, label, idx
        
        return signal, label.long()

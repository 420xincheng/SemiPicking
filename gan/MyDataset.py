import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import maxabs_scale
import os


class GetData:
    def __init__(self, path, signal_size=4096):
        real_classes = [d for d in os.listdir(os.path.join(path, 'real_signal')) if
                        os.path.isfile(os.path.join(path, 'real_signal', d))]
        simulation_classes = [d for d in os.listdir(os.path.join(path, 'simulation_signal')) if
                              os.path.isfile(os.path.join(path, 'simulation_signal', d))]
        real_classes.sort(key=lambda x: int(x[0:-4]))
        simulation_classes.sort(key=lambda x: int(x[0:-4]))
        real_signal = []
        simulation_signal = []
        for file_path in real_classes:
            real_signal.append(np.loadtxt(os.path.join(path, 'real_signal', file_path)))
        for file_path in simulation_classes:
            simulation_signal.append(np.loadtxt(os.path.join(path, 'simulation_signal', file_path)))

        real_signal = np.array(real_signal)
        simulation_signal = np.array(simulation_signal)
        simulation_signal = np.pad(simulation_signal, ((0, 0), (0, signal_size - simulation_signal.shape[-1])),
                                   'constant', constant_values=(0, 0))
        real_signal = maxabs_scale(real_signal, axis=1)
        simulation_signal = maxabs_scale(simulation_signal, axis=1)
        self.real_signal = real_signal[:, np.newaxis, :]
        #         print(self.real_signal)
        self.simulation_signal = simulation_signal[:, np.newaxis, :]

    def get_data(self):
        return self.real_signal, self.simulation_signal


class GetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        return torch.Tensor(data)

    def __len__(self):
        return len(self.data)

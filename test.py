import logging
import sys
from tensorboardX import SummaryWriter
from my_dataset import MyDataset, RandomGenerator
from torchvision import transforms
from models.net import TransUnet
from models.net import CONFIGS
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

DATA_ROOT_PATH = './dataset/'
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Test:
    def __init__(self, data_dir=DATA_ROOT_PATH, test='test',
                 weight_path='./weights/2023-05-14T19-22-10/best_model.pth',
                 batch_size=1, n_gpu=1, out_size=4096):
        self.data_dir = data_dir
        self.mode = test
        self.test_path = os.path.join(DATA_ROOT_PATH, test)
        self.batch_size = batch_size
        self.weight_path = weight_path
        self.out_size = out_size
        assert os.path.exists(self.weight_path), f"file: '{self.weight_path}' dose not exist."

    def test(self, model):
        db_test = MyDataset(data_dir=self.data_dir, data_source='test',
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=self.out_size)]))  # 
        test_loader = DataLoader(db_test, batch_size=self.batch_size, shuffle=False, num_workers=1)

        sm_path = DATA_ROOT_PATH + f'/log/test_log/{TIMESTAMP}'
        os.makedirs(sm_path, exist_ok=True)
        writer = SummaryWriter(sm_path)
        logging.info("{} test iterations per epoch, load model path from {}".format(len(test_loader), self.weight_path))

        model.load_state_dict(torch.load(self.weight_path, map_location=device))
        model.eval()
        test_acc = 0.
        test_num = 0
        with torch.no_grad():
            for id_batch, data in tqdm(enumerate(test_loader)):
                signal, label, case_name = data['signal'], data['label'], data['case_name']
                output, hidden_map, loss_sspcab = model(signal.to(device))
                # out_point = torch.argmax(torch.softmax(output, -1), dim=-1)
                _, out_point = output.max(1)
                print(f"{out_point.item()}, label = {label}, {data['case_name']}")
                test_acc += (out_point == label.to(device)).sum().item()
                test_num += label.size(0)
                logging.info(f' test_acc : {test_acc / (id_batch + 1)}')
                # writer.add_scalar('id_batch/test_acc', test_acc / (id_batch + 1), id_batch)
        test_acc = test_acc / test_num
        print(f'{test_num}, len = {len(test_loader)}')
        logging.info('test_acc : %f%%' % (test_acc * 100))
        return "Testing Finished!"


if __name__ == '__main__':
    SEED = 666
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    log_folder = f'./dataset/log/test_log/{TIMESTAMP}'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + f'/test_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    print(f"using {device} device.")
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f'Using {nw} dataloader workers every process')

    tester = Test(batch_size=1, n_gpu=nw)
    test_net = TransUnet(CONFIGS['TEST_RES'], signal_size=4096, num_classes=1).to(device)
    tester.test(test_net)

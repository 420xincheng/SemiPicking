# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import builtins
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.net import TransUnet
from models.resnet import resnet50_encoder as resnet50
from models.simmatch import get_simmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
from utils.utils import *
from my_dataset import SignalNet

from utils.lr_schedule import warmup_threshold
from datetime import datetime

GPU_ID = 3
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# device = torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch Semi-Picking Net Training')
parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                    help='path to train annotation_x (default: None)')
parser.add_argument('--trainindex_u', default=None, type=str, metavar='PATH',
                    help='path to train annotation_u (default: None)')
parser.add_argument('--arch', metavar='ARCH', default='SimMatch',
                    help='models architecture')
# parser.add_argument('--backbone', default='resnet50_encoder',
#                     choices=backbone_model_names,
#                     help='models architecture: ' +
#                          ' | '.join(backbone_model_names) +
#                          ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=4096, type=int, metavar='N',
                    help='number of classes')
# parser.add_argument('--port', default=23456, type=int, help='dist init port')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=20, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate models on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained models (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained models (default: none)')
parser.add_argument('--super-pretrained', default='', type=str, metavar='PATH',
                    help='path to supervised pretrained models (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--anno-percent', type=float, default=0.1,
                    help='number of labeled data')
parser.add_argument('--split-seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--mu', default=5, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=5, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--threshold', default=0.4, type=float,
                    help='pseudo label threshold')
parser.add_argument('--eman', action='store_true', default=False,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')

parser.add_argument('--models-prefix', default='encoder_q', type=str,
                    help='the models prefix of self-supervised pretrained state_dict')
parser.add_argument('--st', type=float, default=0.2)  # student temperature
parser.add_argument('--tt', type=float, default=0.2)  # teacher temperature
parser.add_argument('--c_smooth', type=float, default=0.9)  # \alpha
parser.add_argument('--DA', default=False, action='store_true')
parser.add_argument('--lambda_in', type=float, default=1)
parser.add_argument('--randaug', default=False, action='store_true')
parser.add_argument('--stack', default=False, action='store_true')
args = parser.parse_args()


def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.to(device)
            target = target.to(device)
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        sum1, cnt1, sum5, cnt5 = utils.torch_dist_sum(args.gpu, top1.sum, top1.count, top5.sum, top5.count)
        top1_acc = sum(sum1.float()) / sum(cnt1.float())
        top5_acc = sum(sum5.float()) / sum(cnt5.float())

    return top1_acc, top5_acc, losses.avg


def main_worker():
    best_acc1 = 0
    best_acc5 = 0

    args.gpu = GPU_ID

    args.distributed = False

    # if rank != 0:
    #     def print_pass(*args):
    #         pass
    #
    #     builtins.print = print_pass

    print(args)

    train_dataset_x, train_dataset_u, val_dataset = SignalNet(mode='train', label=True, return_index=True,corrupt_per=0, cor_num=300), \
                                                    SignalNet(mode='train', label=False), \
                                                    SignalNet(mode='val', label=True),
    # Data loading code
    train_sampler = DistributedSampler

    train_loader_x = DataLoader(
        train_dataset_x,
        batch_size=args.batch_size, shuffle=True, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader_u = DataLoader(
        train_dataset_u,
        batch_size=args.batch_size * args.mu, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=4, shuffle=False, persistent_workers=True,
        num_workers=args.workers, pin_memory=True)

    # create models
    print("=> creating models '{}' with backbone ".format(args.arch))
    model_func = get_simmatch_model(args.arch)
    norm = get_norm(args.norm)
    model = model_func(
        # resnet50,
        'BASELINE',  # Trans
        eman=args.eman,
        momentum=args.ema_m,
        norm=norm,
        K=len(train_dataset_x),
        args=args
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    os.makedirs('./checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/best.pth'
    print(f'checkpoint_path: {checkpoint_path}  , {TIMESTAMP}')
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #     args.start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True
    out_loss = []
    csvDir = f'./dataset/log/train_log/{TIMESTAMP}'
    os.makedirs(csvDir, exist_ok=True)
    if args.evaluate:
        acc1, acc5, val_loss = validate(val_loader, model, criterion, args)
        print('valid * Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))

    else:
        csv_file = open(f'{csvDir}/train_log.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['Epoch', 'totalLoss', 'Loss_x', 'Loss_unlabel', 'Loss_in', 'top1_acc_x', 'top5_acc_x', 'top1_acc_u',
             'top5_acc_u', 'mask', 'pseudoLabel'])

        val_file = open(f'{csvDir}/val_log.csv', 'w', newline='')
        val_writer = csv.writer(val_file)
        val_writer.writerow(
            ['Epoch', 'validLoss', 'top1_acc', 'top5_acc'])

        for epoch in range(args.start_epoch, args.epochs):
            if epoch >= args.warmup_epoch:
                lr_schedule.adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader_x, train_loader_u, model, optimizer, epoch, csv_writer, args)

            if (epoch + 1) % args.eval_freq == 0:
                # evaluate on validation set
                acc1, acc5, val_loss = validate(val_loader, model, criterion, args)
                # remember best acc@1 and save checkpoint
                best_acc1 = max(acc1, best_acc1)
                best_acc5 = max(acc5, best_acc5)
                val_writer.writerow([epoch, val_loss, acc1.item(), acc5.item()])
                print('Valid : Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}' \
                      .format(epoch, acc1, acc5, best_acc1, best_acc5))
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_path)
        csv_file.close()


def train(train_loader_x, train_loader_u, model, optimizer, epoch, csv_writer, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_x = utils.AverageMeter('Loss_x', ':.4e')
    losses_u = utils.AverageMeter('Loss_u', ':.4e')
    losses_in = utils.AverageMeter('Loss_in', ':.4e')
    top1_x = utils.AverageMeter('Acc_x@1', ':6.2f')
    top5_x = utils.AverageMeter('Acc_x@5', ':6.2f')
    top1_u = utils.AverageMeter('Acc_u@1', ':6.2f')
    top5_u = utils.AverageMeter('Acc_u@5', ':6.2f')
    mask_info = utils.AverageMeter('Mask', ':6.3f')
    pseudo_label_info = utils.AverageMeter('Label', ':6.3f')
    curr_lr = utils.InstantMeter('LR', '')
    progress = utils.ProgressMeter(
        len(train_loader_u),
        [curr_lr, batch_time, data_time, losses, losses_x, losses_u, losses_in, top1_x, top5_x, top1_u, top5_u,
         mask_info, pseudo_label_info],
        prefix="Train Epoch: [{}/{}]\t".format(epoch, args.epochs))

    epoch_x = epoch * math.ceil(len(train_loader_u) / len(train_loader_x))
    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch_x))
        train_loader_x.sampler.set_epoch(epoch_x)
        print("set epoch={} for unlabeled sampler".format(epoch))
        train_loader_u.sampler.set_epoch(epoch)

    train_iter_x = iter(train_loader_x)
    # switch to train mode
    model.train()
    if args.eman:
        print("setting the ema models to eval mode")
        if hasattr(model, 'module'):
            model.module.ema.eval()
        else:
            model.ema.eval()

    end = time.time()

    for i, (images_u, targets_u) in enumerate(train_loader_u):
        try:
            images_x, targets_x, index = next(train_iter_x)
        except Exception:
            epoch_x += 1
            print("reshuffle train_loader_x at epoch={}".format(epoch_x))
            if args.distributed:
                print("set epoch={} for labeled sampler".format(epoch_x))
                train_loader_x.sampler.set_epoch(epoch_x)
            train_iter_x = iter(train_loader_x)
            images_x, targets_x, index = next(train_iter_x)

        images_u_w, images_u_s = images_u
        # images_u_w, images_u_s = images_u, images_u
        # measure data loading time
        data_time.update(time.time() - end)

        images_x = images_x.to(device)
        images_u_w = images_u_w.to(device)
        images_u_s = images_u_s.to(device)
        # targets_x, targets_u = targets_x.unsqueeze(-1), targets_u.unsqueeze(-1)  # []
        targets_x = targets_x.to(device)
        targets_u = targets_u.to(device)
        index = index.to(device)

        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(train_loader_u)
            curr_step = epoch * len(train_loader_u) + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        # models forward
        logits_x, pseudo_label, logits_u_s, loss_in = model(images_x, images_u_w, images_u_s, labels=targets_x,
                                                            index=index, start_unlabel=epoch > 0, args=args)
        max_probs, _ = torch.max(pseudo_label, dim=-1)


        # mask = max_probs.ge(args.threshold).float()  # 设定阈值
        mask = max_probs.ge(warmup_threshold(epoch, args)).float()  # 设定变化阈值
        loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
        loss_u = (torch.sum(-F.log_softmax(logits_u_s, dim=1) * pseudo_label.detach(), dim=1) * mask).mean()

        loss_in = loss_in.mean()
        loss = loss_x + args.lambda_u * loss_u + args.lambda_in * loss_in

        # measure accuracy and record loss

        losses.update(loss.item())
        losses_x.update(loss_x.item(), images_x.size(0))
        losses_u.update(loss_u.item(), images_u_w.size(0))
        losses_in.update(loss_in.item(), images_u_w.size(0))
        acc1_x, acc5_x = utils.accuracy(logits_x, targets_x, topk=(1, 5))
        top1_x.update(acc1_x[0], logits_x.size(0))
        top5_x.update(acc5_x[0], logits_x.size(0))
        acc1_u, acc5_u = utils.accuracy(pseudo_label, targets_u, topk=(1, 5))
        top1_u.update(acc1_u[0], pseudo_label.size(0))
        top5_u.update(acc5_u[0], pseudo_label.size(0))
        mask_info.update(mask.mean().item(), mask.size(0))

        bool_mask = mask.bool()
        psudo_label_correct = sum(pseudo_label.max(1)[1][bool_mask] == targets_u[bool_mask]) / (bool_mask.sum() + 1e-8)
        pseudo_label_info.update(psudo_label_correct * 100)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the ema models
        if hasattr(model, 'module'):
            model.module.momentum_update_ema()
        else:
            model.momentum_update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    csv_writer.writerow(
        [epoch, losses.avg, losses_x.avg, losses_u.avg, losses_in.avg, top1_x.avg.item(), top5_x.avg.item(),
         top1_u.avg.item(), top5_u.avg.item(), mask_info.avg,
         pseudo_label_info.avg.item()])
    print(f'max_pro {max_probs.float()} ')


if __name__ == '__main__':
    main_worker()
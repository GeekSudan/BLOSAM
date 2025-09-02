import argparse
import torch
import torch.optim as optim
import os
import sys; sys.path.append("..")
import numpy as np
from model.smooth_cross_entropy import smooth_crossentropy
from torch.optim.lr_scheduler import _LRScheduler
import utility
from utility.step_lr import CosineLR
from utility.log import Log
from utility.initialize import initialize
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.logger import Logger
from model import *
from model.wide_res_net import WideResNet

import torchvision
import torchvision.transforms as transforms
# from blosam import BLOSAM
from optimizers.blosam_2step import BLOSAM
# from timm.models.vision_transformer import VisionTransformer
# import timm


if not os.path.exists('outputs'):
    os.makedirs('outputs')

logger = Logger('outputs/blosam_ds_10_50.txt', title=' ')
logger.set_names(['Valid Loss', 'Valid Acc.', 'Sharpness'])

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_dataloader(name, batch_size=64, workers=8):
        name = name.lower()
        if name == 'fashionmnist':
            transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  
            ])
            trainset = torchvision.datasets.FashionMNIST(
                root='/workspace/dansu/Data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size, shuffle=True, num_workers=workers)

            testset = torchvision.datasets.FashionMNIST(
                root='/workspace/dansu/Data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size, shuffle=False, num_workers=workers)
        elif name.startswith('cifar'):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            if name == 'cifar10':
                trainset = torchvision.datasets.CIFAR10(
                    root='/workspace/dansu/Data', train=True, download=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size, shuffle=True, num_workers=args.workers)

                testset = torchvision.datasets.CIFAR10(
                    root='/workspace/dansu/Data', train=False, download=True, transform=transform_test)
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size, shuffle=False, num_workers=workers)
            elif name == 'cifar100':
                trainset = torchvision.datasets.CIFAR100(
                    root='/workspace/dansu/Data', train=True, download=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size, shuffle=True, num_workers=args.workers)

                testset = torchvision.datasets.CIFAR100(
                    root='/workspace/dansu/Data', train=False, download=True, transform=transform_test)
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size, shuffle=False, num_workers=workers)
            else:
                raise ValueError(f"Unsupport CIFAR variant")

        elif name == 'svhn':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                    std=[0.1980, 0.2010, 0.1970])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                    std=[0.1980, 0.2010, 0.1970])
            ])
            trainset = torchvision.datasets.SVHN(
                root='/workspace/dansu/Data', split='train', download=True, transform=transform_train
            )
            trainloader = torch.utils.data.DataLoader(
                trainset,  batch_size, shuffle=True, num_workers=workers)
            testset = torchvision.datasets.SVHN(
            root='/workspace/dansu/Data', split='test', download=True, transform=transform_test
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=workers
            )
        else:
            raise ValueError(f"Unsupported dataset: {name}")
        return trainloader, testloader

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('--dampening', default=0, type=float, help='dampening')
    parser.add_argument('--arch', '-a', default='ResNet18', type=str)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--p", default=2, type=int, help="Norm")
    parser.add_argument("--xi_lr_ratio", default=50, type=int, help="xi_lr_ratio to lr")
    parser.add_argument('--dataset', default='cifar10', type=str)
    
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloader(args.dataset, args.batch_size, args.workers)

    print("load data success")

    net_name = args.arch
    if args.arch == "r18":
        net = resnet18()
    elif args.arch == "r34":
        net = resnet34()
    elif args.arch == "r34":
        net = resnet34()
    elif args.arch == "r50":
        net = resnet50()
    elif args.arch == "r101":
        net = resnet101()
    elif args.arch == "r152":
        net = resnet152()
    elif args.arch == "m":
        net = mobilenet()
    elif args.arch == "iv3":
        net = inceptionv3()
    elif args.arch == "pr18":
        net = preactresnet18()
    elif args.arch == "pr34":
        net = preactresnet34()
    elif args.arch == "pr50":
        net = preactresnet50()
    elif args.arch == "pr101":
        net = preactresnet101()
    elif args.arch == "pr152":
        net = preactresnet152()
    elif args.arch == "googlenet":
        net = googlenet()
    elif args.arch == "ds":
        net = densenet121()
    elif args.arch == "vgg":
        net = vgg19_bn()
    elif args.arch == "wrn":
        net = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    else:
        raise ValueError("Unknown architecture")
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    # net = PyramidNet(dataset='cifar10', depth=110, alpha=270, num_classes=100, bottleneck=True).to(device)

    # net = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=100).to(device)
    
    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = BLOSAM(net.parameters(),lr=args.learning_rate,rho=args.rho,adaptive=args.adaptive, p=args.p, xi_lr_ratio=args.xi_lr_ratio,momentum_theta=args.momentum, weight_decay=args.weight_decay, dampening=args.dampening) # L2 regularization
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, dampening=args.dampening)
    scheduler = CosineLR(optimizer, args.learning_rate, args.epochs)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)


    best_val_acc = 0.0
    for epoch in range(args.epochs):

        train_losses = AverageMeter('Loss', ':.4e')
        train_top1 = AverageMeter('Acc@1', ':6.2f')
        train_top5 = AverageMeter('Acc@5', ':6.2f') 
        
        test_losses = AverageMeter('Loss', ':.4e') 
        test_top1 = AverageMeter('Acc@1', ':6.2f')
        test_top5 = AverageMeter('Acc@5', ':6.2f')
    
        net.train()
        total_loss_adv = 0.0
        total_loss = 0.0
        for batch_idx, batch in enumerate(trainloader):
            inputs, targets = (b.to(device) for b in batch)
            optimizer.zero_grad()
            # First forward-backward
            enable_running_stats(net)
            predictions = net(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            total_loss += loss.mean().item()
            acc1, acc5 = accuracy(predictions, targets, topk=(1, 5)) 
            train_losses.update(loss.mean(), inputs.size(0))
            train_top1.update(acc1[0], inputs.size(0))
            train_top5.update(acc5[0], inputs.size(0))
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(net)
            # smooth_crossentropy(net(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            loss_adv = smooth_crossentropy(net(inputs), targets, smoothing=args.label_smoothing)
            total_loss_adv += loss_adv.mean().item()
            loss_adv.mean().backward()
            optimizer.second_step(zero_grad=True)
         
            
        avg_loss = total_loss / args.batch_size
        avg_loss_adv = total_loss_adv / args.batch_size
        sharpness = avg_loss_adv - avg_loss
        scheduler(epoch)
    
        print('Epoch {:d} Train Loss {:.4f} Train Acc {:.4f} Sharpness {:.4f}:'.format(epoch, train_losses.avg, train_top1.avg, sharpness))     
        # if epoch > args.warm:
        #     train_scheduler.step()
     
        # train_scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print('Current lr {:.4f}:'.format(current_lr))

        net.eval()
 
        with torch.no_grad():
            for batch in testloader:
                inputs, targets = (b.to(device) for b in batch)
                predictions = net(inputs)
                loss = smooth_crossentropy(predictions, targets)
                acc1, acc5 = accuracy(predictions, targets, topk=(1, 5))
                test_losses.update(loss.mean(), inputs.size(0))
                test_top1.update(acc1[0], inputs.size(0))
            #     test_top5.update(acc5[0], inputs.size(0))
            if test_top1.avg > best_val_acc:
                best_val_acc = test_top1.avg
                torch.save(net.state_dict(), f'/workspace/dansu/BLOSAM/result/best_model_r18_blosam_10.pth')
            print('Epoch {:d} Test Loss {:.4f} Test Acc {:.4f}:'.format(epoch, test_losses.avg, test_top1.avg)) 
            logger.append([f"{test_losses.avg:.4f}", f"{test_top1.avg:.4f}", f"{sharpness:.4f}"])
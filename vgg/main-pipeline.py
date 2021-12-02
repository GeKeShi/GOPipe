import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from vgg_sequential_model import Conv2d, Pool2d, Linear, Flatten
from collections import OrderedDict

import lmdb
from PIL import Image
from PIL import PngImagePlugin
import io

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='VGG16')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--num-split', default=4, type=int,
                    help='number of stages use to train single model')
parser.add_argument('--train-num', default=4, type=int,
                    help='number of minibatch')
parser.add_argument('--recompute', default=False, type=bool,
                    help='recompute for layers except last')
parser.add_argument('--Gpipe', default=False, type=bool,
                    help='Gpipe with DP')
parser.add_argument('--lr_policy', default='polynomial', type=str,
                    help='policy for controlling learning rate, step or polynomial')
parser.add_argument('--lr_warmup', default=True, type=bool,
                    help='lr warmup')
parser.add_argument('--partition', default=3, type=int,
                    help='0:layer-nums 1:time 2:weight 3:HIPPIE 4:manual')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    torch.cuda.set_device(args.gpu)

    rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=rank)

    world_size = args.world_size
    stage_num = args.num_split
    train_num = args.train_num
    num_split = args.num_split
    partition = args.partition
    local_rank = gpu
    group_local_rank = rank % stage_num
    num_worker = world_size // stage_num
    worker_rank = rank // num_split

    # group
    worker_in_group = []
    worker_out_group = []
    worker_group = []

    for i in range(0, num_worker):
        rank_list = []
        if num_split == 2:
            rank_list.append(i * 2)
            rank_list.append(i * 2 + 1)
            group = torch.distributed.new_group(rank_list)
            worker_in_group.append(group)

            rank_list = []
            rank_list.append(i * 2)
            rank_list.append(i * 2 + 1)
            group = torch.distributed.new_group(rank_list)
            worker_in_group.append(group)
        if num_split == 4:
            # 0&1, 1&2, 2&3
            for j in range(0, 3):
                rank_list = []
                rank_list.append(i * 4 + j)
                rank_list.append(i * 4 + j + 1)
                group = torch.distributed.new_group(rank_list)
                worker_in_group.append(group)
            # 0 & 3
            rank_list = []
            rank_list.append(i * 4)
            rank_list.append(i * 4 + 3)
            group = torch.distributed.new_group(rank_list)
            worker_in_group.append(group)
        if num_split == 8:
            # 0&1, 1&2, 2&3, 3&4, 4&5, 5&6, 6&7
            for j in range(0, 7):
                rank_list = []
                rank_list.append(i * 8 + j)
                rank_list.append(i * 8 + j + 1)
                group = torch.distributed.new_group(rank_list)
                worker_in_group.append(group)
            # 0 & 7
            rank_list = []
            rank_list.append(i * 8)
            rank_list.append(i * 8 + 7)
            group = torch.distributed.new_group(rank_list)
            worker_in_group.append(group)

    if num_worker > 1:
        for i in range(0, num_split):
            rank_list = []
            for j in range(0, num_worker):
                rank_list.append(i + j * num_split)
            group = torch.distributed.new_group(rank_list)
            worker_out_group.append(group)

    for i in range(0, num_worker):
        rank_list = []
        for j in range(0, num_split):
            rank_list.append(i * num_split + j)
        worker_group.append(torch.distributed.new_group(rank_list))

    # get group
    if stage_num > 1:
        worker_group = worker_group[worker_rank]
        if group_local_rank == 0:
            worker_in_group_up = worker_in_group[rank + num_split - 1]
            worker_in_group_down = worker_in_group[rank]
        else:
            worker_in_group_up = worker_in_group[rank - 1]
            worker_in_group_down = worker_in_group[rank]
    else:
        worker_group = None
        worker_in_group_up = None
        worker_in_group_down = None

    if num_worker > 1:
        worker_out_group = worker_out_group[group_local_rank]
    else:
        worker_out_group = None

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    cudnn.deterministic = True
    # cudnn.benchmark = False

    # model
    # if stage_num == 1:
    #     model_name = args.arch + '.VGG_v1'
    #     model = eval(model_name)()
    # else:
    #     model_name = args.arch + '.VGG_v' + str(stage_num) + '_' + str(group_local_rank)
    #     model = eval(model_name)()
    if stage_num == 2:
        if group_local_rank == 0:
            if partition == 0:
                model = torch.nn.Sequential(OrderedDict([
                    ('Conv1', Conv2d(3, 64)),
                    ('Conv2', Conv2d(64, 64)),
                    ('Pool1', Pool2d()),

                    ('Conv3', Conv2d(64, 128)),
                    ('Conv4', Conv2d(128, 128)),
                    ('Pool2', Pool2d()),

                    ('Conv5', Conv2d(128, 256)),
                    ('Conv6', Conv2d(256, 256)),
                    ('Conv7', Conv2d(256, 256)),
                    ('Pool3', Pool2d()),

                    ('Conv8', Conv2d(256, 512)),
                    ('Conv9', Conv2d(512, 512)),
                ]))
            if partition == 2:
                model = torch.nn.Sequential(OrderedDict([
                    ('Conv1', Conv2d(3, 64)),
                    ('Conv2', Conv2d(64, 64)),
                    ('Pool1', Pool2d()),

                    ('Conv3', Conv2d(64, 128)),
                    ('Conv4', Conv2d(128, 128)),
                    ('Pool2', Pool2d()),

                    ('Conv5', Conv2d(128, 256)),
                    ('Conv6', Conv2d(256, 256)),
                    ('Conv7', Conv2d(256, 256)),
                    ('Pool3', Pool2d()),

                    ('Conv8', Conv2d(256, 512)),
                    ('Conv9', Conv2d(512, 512)),
                    ('Conv10', Conv2d(512, 512)),
                    ('Pool4', Pool2d()),

                    ('Conv11', Conv2d(512, 512)),
                    ('Conv12', Conv2d(512, 512)),
                    ('Conv13', Conv2d(512, 512)),
                    ('Pool5', nn.AdaptiveAvgPool2d((7, 7))),
                ]))
            if partition == 3:
                model = torch.nn.Sequential(OrderedDict([
                    ('Conv1', Conv2d(3, 64)),
                    ('Conv2', Conv2d(64, 64)),
                    ('Pool1', Pool2d()),

                ]))
            if partition == 4:
                # best
                model = torch.nn.Sequential(OrderedDict([
                    ('Conv1', Conv2d(3, 64)),
                    ('Conv2', Conv2d(64, 64)),
                    ('Pool1', Pool2d()),

                    ('Conv3', Conv2d(64, 128)),
                    ('Conv4', Conv2d(128, 128)),
                    ('Pool2', Pool2d()),
                ]))
        else:
            if partition == 0:
                model = torch.nn.Sequential(OrderedDict([
                    ('Conv10', Conv2d(512, 512)),
                    ('Pool4', Pool2d()),

                    ('Conv11', Conv2d(512, 512)),
                    ('Conv12', Conv2d(512, 512)),
                    ('Conv13', Conv2d(512, 512)),
                    ('Pool5', nn.AdaptiveAvgPool2d((7, 7))),

                    ('Flatten', Flatten()),
                    ('Linear1', Linear(25088, 4096)),
                    ('Dropout1', nn.Dropout(p=0.5)),
                    ('Linear2', Linear(4096, 4096)),
                    ('Dropout2', nn.Dropout(p=0.5)),
                    ('Linear3', Linear(4096, 1000, False, False)),
                ]))
            if partition == 2:
                model = torch.nn.Sequential(OrderedDict([
                    ('Flatten', Flatten()),
                    ('Linear1', Linear(25088, 4096)),
                    ('Dropout1', nn.Dropout(p=0.5)),
                    ('Linear2', Linear(4096, 4096)),
                    ('Dropout2', nn.Dropout(p=0.5)),
                    ('Linear3', Linear(4096, 1000, False, False)),
                ]))
            if partition == 3:
                model = torch.nn.Sequential(OrderedDict([
                    ('Conv3', Conv2d(64, 128)),
                    ('Conv4', Conv2d(128, 128)),
                    ('Pool2', Pool2d()),

                    ('Conv5', Conv2d(128, 256)),
                    ('Conv6', Conv2d(256, 256)),
                    ('Conv7', Conv2d(256, 256)),
                    ('Pool3', Pool2d()),

                    ('Conv8', Conv2d(256, 512)),
                    ('Conv9', Conv2d(512, 512)),
                    ('Conv10', Conv2d(512, 512)),
                    ('Pool4', Pool2d()),

                    ('Conv11', Conv2d(512, 512)),
                    ('Conv12', Conv2d(512, 512)),
                    ('Conv13', Conv2d(512, 512)),
                    ('Pool5', nn.AdaptiveAvgPool2d((7, 7))),

                    ('Flatten', Flatten()),
                    ('Linear1', Linear(25088, 4096)),
                    ('Dropout1', nn.Dropout(p=0.5)),
                    ('Linear2', Linear(4096, 4096)),
                    ('Dropout2', nn.Dropout(p=0.5)),
                    ('Linear3', Linear(4096, 1000, False, False)),
                ]))
            if partition == 4:
                # best
                model = torch.nn.Sequential(OrderedDict([
                    ('Conv5', Conv2d(128, 256)),
                    ('Conv6', Conv2d(256, 256)),
                    ('Conv7', Conv2d(256, 256)),
                    ('Pool3', Pool2d()),

                    ('Conv8', Conv2d(256, 512)),
                    ('Conv9', Conv2d(512, 512)),
                    ('Conv10', Conv2d(512, 512)),
                    ('Pool4', Pool2d()),

                    ('Conv11', Conv2d(512, 512)),
                    ('Conv12', Conv2d(512, 512)),
                    ('Conv13', Conv2d(512, 512)),
                    ('Pool5', nn.AdaptiveAvgPool2d((7, 7))),

                    ('Flatten', Flatten()),
                    ('Linear1', Linear(25088, 4096)),
                    ('Dropout1', nn.Dropout(p=0.5)),
                    ('Linear2', Linear(4096, 4096)),
                    ('Dropout2', nn.Dropout(p=0.5)),
                    ('Linear3', Linear(4096, 1000, False, False)),
                ]))

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # define loss function (criterion) and optimizer
    if group_local_rank == stage_num - 1:
        criterion = nn.CrossEntropyLoss()

    train_len = torch.tensor([0]).cuda()
    val_len = torch.tensor([0]).cuda()

    batch_size = args.batch_size * train_num
    # batch_size = args.batch_size
    if group_local_rank == 0:
        torch.set_num_threads(4)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(valdir,
                                           transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               normalize
                                           ]))

        if num_worker > 1:
            train_sampler = torch.utils.data.distributed. \
                DistributedSampler(train_dataset, num_replicas=num_worker, rank=worker_rank)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        train_len[0] = len(train_loader)
        val_len[0] = len(val_loader)

    dist.broadcast(train_len, 0)
    dist.broadcast(val_len, 0)
    train_len = train_len[0].item()
    val_len = val_len[0].item()

    trainer_options = dict(
        model=model,
        optimizer=optimizer,
        rank=rank,
        group_local_rank=group_local_rank,
        batch_size=batch_size,
        stage_num=stage_num,
        train_num=train_num,
        print_freq=args.print_freq,
        worker_num=num_worker,
        recompute=args.recompute,
        gpipe=args.Gpipe,
        worker_group=worker_group,
        worker_in_group_up=worker_in_group_up,
        worker_in_group_down=worker_in_group_down,
        worker_out_group=worker_out_group
        )
    if group_local_rank == stage_num - 1:
        trainer_options['criterion'] = criterion
    trainer = Trainer(**trainer_options)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(trainer.optimizer, epoch, args)
        is_best = torch.tensor([0]).cuda()
        if num_worker > 1 and group_local_rank == 0:
            train_sampler.set_epoch(epoch)
        if stage_num == 1:
            trainer.feed_data(train_loader, epoch, True)
        else:
            if group_local_rank == 0:
                trainer.feed_data_stage0(train_loader, epoch, True, args.lr)
                trainer.barrier()
                trainer.feed_data_stage0(val_loader, epoch, False, args.lr)
                dist.broadcast(is_best, stage_num - 1)
            elif group_local_rank == stage_num - 1:
                acc = trainer.feed_data_last(train_len, epoch, True, args.lr)
                trainer.barrier()
                acc1 = trainer.feed_data_last(val_len, epoch, False, args.lr)
                # remember best acc@1 and save checkpoint
                is_best[0] = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                dist.broadcast(is_best, stage_num - 1)
            else:
                trainer.feed_data_rest(train_len, epoch, True, args.lr)
                trainer.barrier()
                trainer.feed_data_rest(val_len, epoch, False, args.lr)
                dist.broadcast(is_best, stage_num - 1)
        if rank < stage_num:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'VGG16',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, 'stage' + str(rank))


def lmdb_loader(path, lmdb_data):
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(root, transform=None, target_transform=None, loader=lmdb_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    print('Loading pt{} and lmdb {}'.format(pt_path, lmdb_path))
    data_set = torch.load(pt_path)
    data_set.lmdb_data = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 rank,
                 group_local_rank,
                 batch_size,
                 print_freq,
                 stage_num=4,
                 train_num=4,
                 worker_num=1,
                 recompute=False,
                 criterion=None,
                 gpipe=False,
                 worker_group=None,
                 worker_in_group_up=None,
                 worker_in_group_down=None,
                 worker_out_group=None):
        super(Trainer, self).__init__()
        self.model = model.cuda()
        self.optimizer = optimizer
        self.rank = rank
        self.group_local_rank = group_local_rank
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.stage_num = stage_num
        self.train_num = train_num
        self.worker_num = worker_num
        self.recompute = recompute
        if criterion is not None:
            self.criterion = criterion.cuda()
        self.worker_group = worker_group
        self.worker_in_group_up = worker_in_group_up
        self.worker_in_group_down = worker_in_group_down
        self.worker_out_group = worker_out_group
        self.Grad = []
        self.recv = None
        self.out_grad = None
        self.gpipe = gpipe

        self.handle = None
        self.shape = []
        self.split_size = []
        self.grad_flatten = None

        if gpipe:
            self.recompute = False

        for p in self.model.parameters():
            if p.requires_grad:
                self.get_shape(p)
                if self.group_local_rank < stage_num // 2 and self.worker_num > 1:
                    self.Grad.append(torch.zeros(p.shape).cuda())

    def get_shape(self, p):
        shape_ = []
        dim = len(p.shape)
        num = 1
        for k in range(dim):
            num = num * p.shape[k]
            shape_.append(p.shape[k])
        self.split_size.append(num)
        self.shape.append(shape_)

    def barrier(self, group=None):
        if group is None:
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        else:
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1), group=group)
        torch.cuda.synchronize()

    def grad_hook(self, grad):
        # self.grad_up = grad.detach()
        dist.broadcast(grad.detach(), self.rank,
                       group=self.worker_in_group_up)

    # all_reduce
    def all_reduce_0(self):
        if self.worker_num > 1:
            for k in range(len(self.Grad)):
                self.Grad[k] = self.Grad[k].flatten()
            self.Grad = torch.cat(self.Grad)
            self.Grad = torch.true_divide(self.Grad, self.train_num * self.worker_num)
            torch.cuda.synchronize()
            self.handle = dist.all_reduce(self.Grad, op=dist.ReduceOp.SUM,
                                          group=self.worker_out_group, async_op=True)

    def all_reduce_1(self):
        if self.worker_num > 1:
            self.Grad = []
            for k, params in enumerate(self.model.parameters()):
                self.Grad.append(params.grad.detach().clone().flatten())
            self.Grad = torch.cat(self.Grad)
            self.Grad = torch.true_divide(self.Grad, self.train_num * self.worker_num)
            dist.all_reduce(self.Grad, op=dist.ReduceOp.SUM, group=self.worker_out_group)
            torch.cuda.synchronize()
            # XXX
            self.Grad = list(self.Grad.split(self.split_size))
            for k, params in enumerate(self.model.parameters()):
                if params.requires_grad:
                    self.Grad[k] = self.Grad[k].resize_(self.shape[k])
                    params.grad = self.Grad[k]

    # update
    def update_0(self):
        if self.worker_num > 1:
            self.handle.wait()
            # XXX
            self.Grad = list(self.Grad.split(self.split_size))
            for k, params in enumerate(self.model.parameters()):
                if params.requires_grad:
                    self.Grad[k] = self.Grad[k].resize_(self.shape[k])
                    grad = params.grad.detach().clone()
                    params.grad = self.Grad[k]
                    self.Grad[k] = grad
        if self.worker_num == 1:
            for params in self.model.parameters():
                if params.grad is not None:
                    params.grad.data = torch.true_divide(params.grad.data, self.train_num)
        self.optimizer.step()
        self.model.zero_grad()

    def update_1(self):
        if self.worker_num == 1:
            for params in self.model.parameters():
                if params.grad is not None:
                    params.grad.data = torch.true_divide(params.grad.data, self.train_num)
        self.optimizer.step()
        self.model.zero_grad()

    # feed_date
    def feed_data(self, train_loader, epoch, training):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        throughout = AverageMeter('throughout', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5, throughout],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        if training:
            self.model.train()
        else:
            self.model.eval()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if training:
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.all_reduce_1()
                self.update_1()

            # measure elapsed time
            throughout.update(images.size(0) / (time.time() - end))
            batch_time.update(time.time() - end)
            end = time.time()

            if i == 0:
                torch.cuda.empty_cache()
            if i % self.print_freq == 0:
                progress.display(i)

    def feed_data_stage0(self, train_loader, epoch, training, base_lr):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        throughout = AverageMeter('throughout', ':6.3f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, throughout],
            prefix="Epoch: [{}]".format(epoch))
        images = torch.randn(train_loader.batch_size, 3, 224, 224).cuda()
        self.pre_stage0(images)
        if training:
            self.model.train()
            end = time.time()
            for i, (images, target) in enumerate(train_loader):
                if i == len(train_loader) - 1:
                    continue
                # measure data loading time
                data_time.update(time.time() - end)
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                self.iterate_stage0(images, target, training)
                # measure elapsed time
                throughout.update(images.size(0)/(time.time()-end))
                batch_time.update(time.time() - end)
                end = time.time()
                # time.sleep(20)
                # if epoch < 5:
                #     warmup(self.optimizer, i, epoch, len(train_loader), base_lr)
                if i % 100 == 0:
                    print("lr=", self.optimizer.param_groups[0]['lr'])
                if i == 0:
                    torch.cuda.empty_cache()
                if i % self.print_freq == 0:
                    progress.display(i)
        else:
            self.model.eval()
            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(train_loader):
                    if i == len(train_loader) - 1:
                        continue
                    # measure data loading time
                    data_time.update(time.time() - end)
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    self.iterate_stage0(images, target, training)
                    # measure elapsed time
                    throughout.update(images.size(0) / (time.time() - end))
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % self.print_freq == 0:
                        progress.display(i)

    def feed_data_rest(self, train_len, epoch, training, base_lr):
        self.pre_rest()
        if training:
            self.model.train()
            for i in range(0, train_len):
                if i == train_len - 1:
                    continue
                self.iterate_rest(training)
                if i == 0:
                    torch.cuda.empty_cache()
                # if epoch < 5:
                #     warmup(self.optimizer, i, epoch, train_len, base_lr)

        else:
            self.model.eval()
            with torch.no_grad():
                for i in range(0, train_len):
                    if i == train_len - 1:
                        continue
                    self.iterate_rest(training)

    def feed_data_last(self, train_len, epoch, training, base_lr):
        # print("epoch", epoch)
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            train_len,
            [losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        self.pre_last()
        if training:
            self.model.train()
            for i in range(0, train_len):
                if i == train_len - 1:
                    continue
                losses, top1, top5 = self.iterate_last(training, losses, top1, top5)
                # if epoch < 5:
                #     warmup(self.optimizer, i, epoch, train_len, base_lr)
                if i == 0:
                    torch.cuda.empty_cache()
                if i % self.print_freq == 0:
                    progress.display(i)
                if not self.gpipe and self.rank == self.stage_num - 1:
                    if i % 500 == 99:
                        print_loss = format(losses.avg, '.4f')
                        print_time = format(time.time(), '.4f')
                        print_acc1 = format(top1.avg, '.4f')
                        with open('./VGG-results/pipeline-loss.txt', 'a') as file:
                            file.write(str(epoch) + '-' + str(epoch * train_len + i) +
                                       '-' + str(print_time) + '-' + str(print_loss) +
                                       '-' + str(print_acc1) + '\n')
        else:
            self.model.eval()
            for i in range(0, train_len):
                if i == train_len - 1:
                    continue
                losses, top1, top5 = self.iterate_last(training, losses, top1, top5)
                if i % self.print_freq == 0:
                    progress.display(i)
            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            if not self.gpipe and self.rank == self.stage_num - 1:
                print_acc1 = format(top1.avg, '.4f')
                print_acc5 = format(top5.avg, '.4f')
                print_time = format(time.time(), '.4f')
                with open('./VGG-results/pipeline-acc.txt', 'a') as file:
                    file.write(str(epoch) + '-' + str(print_time) +
                               '-' + str(print_acc1) + '-' + str(print_acc5) + '\n')
        return top1.avg

    def pre_stage0(self, images):
        images_split = images.chunk(self.train_num)
        out = self.model(images_split[0])
        dist.broadcast(torch.tensor(out.shape).int().cuda(),
                       self.rank, group=self.worker_in_group_down)
        self.out_grad = torch.zeros_like(out).cuda()
        del out

    def pre_rest(self):
        shape_ = torch.zeros((4)).int().cuda()
        dist.broadcast(shape_,
                       self.rank - 1, group=self.worker_in_group_up)
        self.recv = torch.randn((shape_[0].item(), shape_[1].item(),
                                 shape_[2].item(), shape_[3].item())).cuda()
        out = self.model(self.recv)
        dist.broadcast(torch.tensor(out.shape).int().cuda(),
                       self.rank, group=self.worker_in_group_down)
        self.out_grad = torch.zeros_like(out).cuda()
        del out

    def pre_last(self):
        shape_ = torch.zeros((4)).int().cuda()
        dist.broadcast(shape_,
                       self.rank - 1, group=self.worker_in_group_up)
        self.recv = torch.randn((shape_[0].item(), shape_[1].item(),
                                 shape_[2].item(), shape_[3].item())).cuda()
        self.target = torch.zeros(self.batch_size).long().cuda()

    # iterate
    def iterate_stage0(self, images, target, training):
        dist.broadcast(target, self.rank, group=self.worker_in_group_up)
        images_split = list(images.chunk(self.train_num))
        out_list = []
        # forward
        for k in range(0, self.train_num):
            images = images_split[k]
            out = self.model(images)
            if k == self.stage_num - 1 and training:
                self.all_reduce_0()

            if k >= self.stage_num - 1 and training:
                if self.recompute:
                    out_back = self.model(images_split[k - self.stage_num + 1])
                else:
                    out_back = out_list.pop(0)
                self.out_grad = self.out_grad * 0
                dist.broadcast(self.out_grad, self.rank + 1,
                               group=self.worker_in_group_down)
                # pipeline_send
                dist.broadcast(out.detach(),
                               self.rank, group=self.worker_in_group_down)
                # backward
                torch.cuda.synchronize()
                # out_back.backward(self.out_grad)
                torch.autograd.backward((out_back, ), (self.out_grad, ))
            else:
                # pipeline_send
                dist.broadcast(out.detach(),
                               self.rank, group=self.worker_in_group_down)
            if self.recompute:
                del out
            elif training:
                out_list.append(out)

        # backward
        if training:
            for k in range(0, self.stage_num - 1):
                if self.recompute:
                    out_back = self.model(images_split.pop(0))
                else:
                    out_back = out_list.pop(0)
                self.out_grad = self.out_grad * 0
                dist.broadcast(self.out_grad, self.rank + 1,
                               group=self.worker_in_group_down)
                # backward
                # out_back.backward(self.out_grad)
                torch.autograd.backward((out_back,), (self.out_grad,))
                torch.cuda.synchronize()
            self.update_0()

    def iterate_rest(self, training):
        inputs_list = []
        out_list = []
        # forward
        for k in range(0, self.train_num):
            # receive
            dist.broadcast(self.recv, self.rank - 1, group=self.worker_in_group_up)
            inputs = self.recv.detach().clone()
            inputs.requires_grad = True
            inputs.retain_grad()
            inputs.register_hook(self.grad_hook)
            out = self.model(inputs)

            # XXX wait()
            if k == self.stage_num - self.group_local_rank - 1:
                if training and self.group_local_rank < self.stage_num // 2 and self.worker_num > 1:
                    self.all_reduce_0()

            if k >= self.stage_num - self.group_local_rank - 1 and training:
                if self.recompute:
                    out_back = self.model(inputs_list.pop(0))
                else:
                    out_back = out_list.pop(0)
                self.out_grad = self.out_grad * 0
                dist.broadcast(self.out_grad, self.rank + 1,
                               group=self.worker_in_group_down)
                # pipeline_send
                dist.broadcast(out.detach(),
                               self.rank, group=self.worker_in_group_down)
                # backward
                torch.cuda.synchronize()
                out_back.backward(self.out_grad)
            else:
                # pipeline_send
                dist.broadcast(out.detach(),
                               self.rank, group=self.worker_in_group_down)
            if self.recompute:
                inputs_list.append(inputs)
                del out
            elif training:
                out_list.append(out)

        # backward
        if training:
            for k in range(0, self.stage_num - 1):
                if self.recompute:
                    out_back = self.model(images_split.pop(0))
                else:
                    out_back = out_list.pop(0)
                self.out_grad = self.out_grad * 0
                dist.broadcast(self.out_grad, self.rank + 1,
                               group=self.worker_in_group_down)
                # backward
                out_back.backward(self.out_grad)
                torch.cuda.synchronize()
            self.update_0()

    def iterate_last(self, training, losses, top1, top5):
        dist.broadcast(self.target, self.rank - self.group_local_rank,
                       group=self.worker_in_group_down)
        target_split = self.target.chunk(self.train_num)

        for k in range(0, self.train_num):
            # forward
            target = target_split[k]
            # pipeline_recv
            self.recv = self.recv * 0
            dist.broadcast(self.recv, self.rank - 1, group=self.worker_in_group_up)
            # forward
            inputs = self.recv.detach()
            inputs.requires_grad = True
            inputs.retain_grad()
            inputs.register_hook(self.grad_hook)
            output = self.model(inputs)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))
            loss.backward()

        if training:
            self.all_reduce_1()
            self.update_1()
        return losses, top1, top5


def save_checkpoint(state, is_best, stage, filename='checkpoint.pth.tar'):
    filename = stage + '_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, stage + '_' + 'model_best.pth.tar')


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def warmup(optimizer, step, epoch, epoch_length, base_lr):
    lr = base_lr * float(1 + step + epoch * epoch_length)/(5.0*epoch_length)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # if not args.lr_warmup or epoch > 4:
    #     if args.lr_policy == "polynomial":
    #         power = 2.0
    #         lr = args.lr * ((1.0 - (float(epoch) / float(args.epochs))) ** power)
    #     elif args.lr_policy == 'step':
    #         lr = args.lr * (0.1 ** (epoch // 20))
    #     else:
    #         raise NotImplementedError
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

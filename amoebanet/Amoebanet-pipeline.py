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

from amoebanet import amoebanetd, amoebanetd_pipeline
from torch.optim import SGD
import torch.nn.functional as F
from collections import OrderedDict
from typing import cast, List, Iterable, Any, Optional, Tuple, Union

# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim
# import torch.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--num-layers', default=18, type=int)
parser.add_argument('--num-filters', default=208, type=int)
parser.add_argument('--num-split', default=4, type=int,
                    help='number of stages use to train single model')
parser.add_argument('--train-num', default=4, type=int,
                    help='number of minibatch')
parser.add_argument('--recompute', default=False, type=bool,
                    help='recompute for layers except last')
parser.add_argument('--Gpipe', default=False, type=bool,
                    help='recompute for layers except last')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--partition', default=2, type=int,
                    help='0:layer-nums 1:time 2:weight 3:HIPPIE')


def split_module(module, balance, devices):
    """Splits a module into multiple partitions.

    Returns:
        A tuple of (partitions, balance, devices).

        Partitions are represented as a :class:`~torch.nn.ModuleList` whose
        item is a partition. All layers in a partition are placed in the
        same device.

    Raises:
        BalanceError:
            wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    balance = list(balance)

    if len(module) != sum(balance):
        raise Exception('module and sum of balance have different length '
                        f'(module: {len(module)}, sum of balance: {sum(balance)})')

    if any(x <= 0 for x in balance):
        raise Exception(f'all balance numbers must be positive integer (balance: {balance})')

    # if len(balance) > len(devices):
    #     raise IndexError('too few devices to hold given partitions '
    #                      f'(devices: {len(devices)}, partitions: {len(balance)})')

    j = 0
    partitions = []
    layers = OrderedDict()

    for name, layer in module.named_children():
        layers[name] = layer

        if len(layers) == balance[j]:
            # Group buffered layers as a partition.
            partition = nn.Sequential(layers)

            # device = devices[j]
            # partition.to(device)

            partitions.append(partition)

            # Prepare for the next partition.
            layers.clear()
            j += 1

    partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
    # del devices[j:]

    # return partitions, balance, devices
    return partitions


def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    torch.cuda.set_device(args.gpu)

    rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=rank)

    world_size = args.world_size
    # stage_num = args.num_split
    train_num = args.train_num
    num_split = args.num_split
    local_rank = gpu
    group_local_rank = rank % num_split
    num_worker = world_size // num_split
    worker_rank = rank // num_split

    batch_size = args.batch_size * train_num
    num_filters = args.num_filters

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
    if num_split > 1:
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
    torch.set_num_threads(4)

    model = amoebanetd(num_classes=1000, num_layers=args.num_layers,
                       num_filters=args.num_filters)
                       
    if rank == 0:
        print("model", model)
    
    if args.num_layers == 18:
        # 24 num
        if num_split == 4:
            balance = [3, 6, 8, 7]
        if num_split == 2:
            if args.partition == 0:
                balance = [12, 12]
            if args.partition == 2:
                # weight
                balance = [19, 5]
            if args.partition == 3:
                # hippie
                balance = [9, 15]
    elif args.num_layers == 36:
        if num_split == 4:
            balance = [16, 12, 7, 7]
        if num_split == 2:
            balance = [28, 14]
    model_list = split_module(model, balance, [])

    # model
    model = model_list[group_local_rank].cuda()
    param_memory = sum(p.numel() for p in model.parameters())
    print(group_local_rank, round(param_memory * 4 / (1024 * 1024), 2))
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    dataset_size = 1000

    inputs = torch.rand(batch_size, 3, 224, 224).cuda()
    target = torch.randint(1000, (batch_size,)).long().cuda()

    trainer_options = dict(
        model=model,
        optimizer=optimizer,
        rank=rank,
        group_local_rank=group_local_rank,
        batch_size=batch_size,
        num_layers=args.num_layers,
        num_filters=num_filters,
        stage_num=num_split,
        train_num=train_num,
        worker_num=num_worker,
        recompute=args.recompute,
        gpipe=args.Gpipe,
        worker_group=worker_group,
        worker_in_group_up=worker_in_group_up,
        worker_in_group_down=worker_in_group_down,
        worker_out_group=worker_out_group
    )

    trainer = Trainer(**trainer_options)

    epochs = 30
    for epoch in range(epochs):
        print("epoch", epoch)
        if group_local_rank == 0:
            trainer.feed_data_stage0(inputs)
        elif group_local_rank == num_split - 1:
            trainer.feed_data_last(target)
        else:
            trainer.feed_data_rest()


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 rank,
                 group_local_rank,
                 batch_size,
                 num_layers,
                 num_filters,
                 print_freq=10,
                 data_len=1000,
                 stage_num=4,
                 train_num=4,
                 worker_num=1,
                 recompute=False,
                 gpipe=False,
                 worker_group=None,
                 worker_in_group_up=None,
                 worker_in_group_down=None,
                 worker_out_group=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
        self.group_local_rank = group_local_rank
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.stage_num = stage_num
        self.train_num = train_num
        self.worker_num = worker_num
        self.recompute = recompute
        self.gpipe = gpipe
        # self.criterion = criterion
        self.worker_group = worker_group
        self.worker_in_group_up = worker_in_group_up
        self.worker_in_group_down = worker_in_group_down
        self.worker_out_group = worker_out_group
        self.data_len = data_len
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.inp_recv = []
        self.grad_recv = []
        self.split_size = []
        self.shape = []
        self.handle = None
        self.Grad = []
        if self.gpipe:
            self.recompute = False
        # grad
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

    def pipeline_send(self, out):
        if isinstance(out, torch.Tensor):
            out = (out,)
        for tensor in out:
            dist.broadcast(tensor.detach().clone(), self.rank, group=self.worker_in_group_down)
            torch.cuda.synchronize()

    def pipeline_recv(self):
        length = len(self.inp_recv)
        if length == 1:
            dist.broadcast(self.inp_recv[0], self.rank - 1, group=self.worker_in_group_up)
            inputs = self.inp_recv[0]
            inputs.requires_grad = True
            inputs.retain_grad()
        else:
            inputs = []
            for inp in self.inp_recv:
                dist.broadcast(inp, self.rank - 1, group=self.worker_in_group_up)
                inp.requires_grad = True
                inp.retain_grad()
                inputs.append(inp)
            inputs = tuple(inputs)
        return inputs

    def grad_se(self, inp):
        if isinstance(inp, torch.Tensor):
            inp = (inp,)
        for tensor in inp:
            dist.broadcast(tensor.grad.detach(), self.rank, group=self.worker_in_group_up)

    def grad_re(self, out):
        if isinstance(out, torch.Tensor):
            dist.broadcast(self.grad_recv[0], self.rank + 1, group=self.worker_in_group_down)
            grad_tensors = (self.grad_recv[0], )
            out_tensors = (out, )
        if isinstance(out, tuple):
            grad_tensors = []
            for grad in self.grad_recv:
                dist.broadcast(grad, self.rank + 1, group=self.worker_in_group_down)
                grad_tensors.append(grad)
            grad_tensors = tuple(grad_tensors)
            out_tensors = out
        assert len(out_tensors) == len(grad_tensors)
        return out_tensors, grad_tensors

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

    def feed_data_stage0(self, images):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        throughout = AverageMeter('throughout', ':6.3f')
        progress = ProgressMeter(
            self.data_len,
            [batch_time, data_time, throughout])
        self.pre_stage0(images)
        end = time.time()
        for i in range(self.data_len):
            images = images.cuda()
            self.iterate_stage0(images)
            # measure elapsed time
            throughout.update(images.size(0) / (time.time() - end))
            batch_time.update(time.time() - end)
            # time.sleep(20)
            if i == 0:
                torch.cuda.empty_cache()
            if i % self.print_freq == 0:
                progress.display(i)
            end = time.time()

    def feed_data_rest(self):
        self.pre_rest()
        for i in range(self.data_len):
            self.iterate_rest()
            if i == 0:
                torch.cuda.empty_cache()

    def feed_data_last(self, target):
        self.pre_last()
        for i in range(self.data_len):
            self.iterate_last(target)
            if i == 0:
                torch.cuda.empty_cache()

    def pre_stage0(self, images):
        images_split = images.chunk(self.train_num)
        out = self.model(images_split[0])
        if isinstance(out, torch.Tensor):
            out = (out,)
        num_send = torch.Tensor([len(out)]).int().cuda()
        print("num", num_send)
        dist.broadcast(num_send, self.rank, group=self.worker_in_group_down)
        for tensor in out:
            shape = torch.Tensor(list(tensor.shape)).int().cuda()
            print("shape", shape)
            dist.broadcast(shape, self.rank, group=self.worker_in_group_down)
            grad = torch.zeros_like(tensor).cuda()
            self.grad_recv.append(grad)
        del out, num_send, grad, shape

    def pre_rest(self):
        num_recv = torch.Tensor([0]).int().cuda()
        dist.broadcast(num_recv, self.rank - 1, group=self.worker_in_group_up)
        for i in range(0, num_recv.item()):
            shape = torch.Tensor([0, 0, 0, 0]).int().cuda()
            dist.broadcast(shape, self.rank - 1, group=self.worker_in_group_up)
            inp = torch.zeros(shape.tolist()).cuda()
            self.inp_recv.append(inp)

        out = self.model(self.inp_recv)
        if isinstance(out, torch.Tensor):
            out = (out,)
        num_send = torch.Tensor([len(out)]).int().cuda()
        dist.broadcast(num_send, self.rank, group=self.worker_in_group_down)
        for tensor in out:
            shape = torch.Tensor(list(tensor.shape)).int().cuda()
            dist.broadcast(shape, self.rank, group=self.worker_in_group_down)
            grad = torch.zeros_like(tensor).cuda()
            self.grad_recv.append(grad)
        del out, num_recv, num_send, grad, shape, inp

    def pre_last(self):
        num_recv = torch.Tensor([0]).int().cuda()
        dist.broadcast(num_recv, self.rank - 1, group=self.worker_in_group_up)
        for i in range(0, num_recv.item()):
            shape = torch.Tensor([0, 0, 0, 0]).int().cuda()
            dist.broadcast(shape, self.rank - 1, group=self.worker_in_group_up)
            inp = torch.zeros(shape.tolist()).cuda()
            self.inp_recv.append(inp)
        del num_recv, shape, inp

    # iterate
    def iterate_stage0(self, images):
        training = True
        images_split = images.chunk(self.train_num)
        out_list = []
        # forward
        for k in range(0, self.train_num):
            images = images_split[k]
            out = self.model(images)
            out_list.append(out)

            if k == self.stage_num - 1:
                self.all_reduce_0()

            if k >= self.stage_num - 1 and training:
                # grad_recv
                out_back = out_list.pop(0)
                out_tensors, grad_tensors = self.grad_re(out_back)
                # pipeline_send
                self.pipeline_send(out)
                # backward
                torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
                torch.cuda.synchronize()

            else:
                # pipeline_send
                self.pipeline_send(out)

        # backward
        for k in range(0, self.stage_num - 1):
            # grad_recv
            out_back = out_list.pop(0)
            out_tensors, grad_tensors = self.grad_re(out_back)
            # backward
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
            torch.cuda.synchronize()
        self.update_0()

    def iterate_rest(self):
        inp_list = []
        out_list = []
        for k in range(0, self.train_num):
            # pipeline_recv
            inp = self.pipeline_recv()
            out = self.model(inp)
            torch.cuda.synchronize()
            inp_list.append(inp)
            out_list.append(out)

            # XXX wait()
            if k == self.stage_num - self.group_local_rank - 1:
                if self.group_local_rank < self.stage_num // 2 and self.worker_num > 1:
                    self.all_reduce_0()

            if k >= self.stage_num - self.group_local_rank - 1:
                # grad_recv
                out_back = out_list.pop(0)
                out_tensors, grad_tensors = self.grad_re(out_back)
                # pipeline_send
                self.pipeline_send(out)
                # backward
                torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
                torch.cuda.synchronize()
                # grad_send
                inp_back = inp_list.pop(0)
                self.grad_se(inp_back)
            else:
                # pipeline_send
                self.pipeline_send(out)
        # backward
        for k in range(0, self.stage_num - self.group_local_rank - 1):
            # grad_recv
            out_back = out_list.pop(0)
            out_tensors, grad_tensors = self.grad_re(out_back)
            # backward
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
            torch.cuda.synchronize()
            # grad_send
            inp_back = inp_list.pop(0)
            self.grad_se(inp_back)
        # update
        if self.group_local_rank < self.stage_num // 2 and self.worker_num > 1:
            self.update_0()
        else:
            self.all_reduce_1()
            self.update_1()

    def iterate_last(self, target):
        target_split = target.chunk(self.train_num)
        for k in range(0, self.train_num):
            # forward
            target = target_split[k]
            # pipeline_recv
            inp = self.pipeline_recv()
            out = self.model(inp)
            loss = F.cross_entropy(out, target)

            # backward
            loss.backward()
            self.grad_se(inp)
            torch.cuda.synchronize()
        self.all_reduce_1()
        self.update_1()


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

        # if self.train_num == 1:
        #     target = target_split[0]
        #     dist.broadcast(recv, self.rank - 1, group=self.worker_in_group_up)
        #     recv_list.append(recv.detach().clone())
        #
        #     # forward
        #     inputs = recv.detach()
        #     inputs.requires_grad = True
        #     inputs.retain_grad()
        #     output = self.model(inputs)
        #     loss = self.criterion(output, target)
        #
        #     # measure accuracy and record loss
        #     acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #     losses.update(loss.item(), target.size(0))
        #     top1.update(acc1[0], target.size(0))
        #     top5.update(acc5[0], target.size(0))
        #
        #     loss.backward()
        #     torch.cuda.synchronize()
        #     dist.broadcast(inputs.grad.detach().clone(),
        #                    self.rank, group=self.worker_in_group_up)
        # else:
        #     for k in range(0, self.train_num):
        #         target = target_split[k]
        #         # receive
        #         if k < self.train_num // 2:
        #             dist.broadcast(recv, self.rank - 1, group=self.worker_in_group_up)
        #             recv_list.append(recv.detach().clone())
        #
        #             # forward
        #             inputs = recv_list[k]
        #             inputs.requires_grad = True
        #             inputs.retain_grad()
        #             output = self.model(inputs)
        #             loss = self.criterion(output, target)
        #
        #             # measure accuracy and record loss
        #             acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #             losses.update(loss.item(), target.size(0))
        #             top1.update(acc1[0], target.size(0))
        #             top5.update(acc5[0], target.size(0))
        #
        #             # receive
        #             dist.broadcast(recv, self.rank - 1, group=self.worker_in_group_up)
        #             recv_list.append(recv.detach().clone())
        #
        #         else:
        #             # forward
        #             inputs = recv_list[k]
        #             inputs.requires_grad = True
        #             inputs.retain_grad()
        #             output = self.model(inputs)
        #             loss = self.criterion(output, target)
        #
        #             # measure accuracy and record loss
        #             acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #             losses.update(loss.item(), target.size(0))
        #             top1.update(acc1[0], target.size(0))
        #             top5.update(acc5[0], target.size(0))
        #
        #         if training:
        #             # backward
        #             if k > self.train_num // 2 - 1:
        #                 dist.broadcast(grad_list[2 * (k - self.train_num // 2)],
        #                                self.rank, group=self.worker_in_group_up)
        #                 loss.backward()
        #                 torch.cuda.synchronize()
        #                 grad_list.append(inputs.grad.detach().clone())
        #                 dist.broadcast(grad_list[2 * (k - self.train_num // 2) + 1],
        #                                self.rank, group=self.worker_in_group_up)
        #             else:
        #                 loss.backward()
        #                 torch.cuda.synchronize()
        #                 grad_list.append(inputs.grad.detach().clone())
        # # update
        # if training:
        #     # all_reduce
        #     self.all_reduce_1()
        #     self.update_1()

    # args = parser.parse_args()
    # model = amoebanetd(num_classes=1000, num_layers=args.num_layers,
    #                    num_filters=args.num_filters)
    # balance = [3, 6, 8, 7]
    # model_list = split_module(model, balance, [])
    # # print("model_list", type(model_list))
    # # print("model1", type(model_list[0]))
    # model1 = model_list[0].cuda(0)
    # model2 = model_list[1].cuda(1)
    # model3 = model_list[2].cuda(2)
    # model4 = model_list[3].cuda(3)
    # optimizer1 = SGD(model1.parameters(), lr=0.1)
    # optimizer2 = SGD(model2.parameters(), lr=0.1)
    # optimizer3 = SGD(model3.parameters(), lr=0.1)
    # optimizer4 = SGD(model4.parameters(), lr=0.1)
    # print("finish model")
    # # for name, params in model.named_parameters():
    # #     print(name, ":", params.size())
    # time.sleep(3)
    # batch_size = args.batch_size
    # num_filters = args.num_filters
    # dataset_size = 1000
    # inputs = torch.rand(batch_size, 3, 224, 224).cuda(0)
    # target = torch.randint(1000, (batch_size, )).cuda(3)
    # epochs = args.epochs
    # for epoch in range(epochs):
    #     print("epoch")
    #     time1 = 0.0
    #     time2 = 0.0
    #     time3 = 0.0
    #     time4 = 0.0
    #     data_len = 0
    #     losses = 0.0
    #     for i in range(dataset_size):
    #         data_len += inputs.shape[0]
    #         start = time.time()
    #         out1 = model1(inputs)
    #         # print("out1", type(out1), len(out1), type(out1[0]), type(out1[1]))
    #         # print("out1", out1[0].size(), out1[1].size())
    #         # print("out1", type(out1), len(out1))
    #         # out_0, out_1 = out1[0].detach().clone().cuda(1), out1[1].detach().clone().cuda(1)
    #         out_0, out_1 = out1
    #         out_0 = torch.cat((out_0, out_0), dim=2)
    #         out_0 = torch.cat((out_0, out_0), dim=3)
    #         # print("out0", out_0.shape)
    #         out1 = torch.cat((out_0, out_1), dim=1)
    #         time1 += time.time() - start
    #         inputs2 = out1.detach().clone().cuda(1)
    #         inputs2.requires_grad = True
    #         inputs2.retain_grad()
    #         # out_0.requires_grad = True
    #         # out_0.retain_grad()
    #         # out_1.requires_grad = True
    #         # out_1.retain_grad()
    #         # print("1", out_0.shape, out_1.shape)
    #         inputs2_ = (inputs2[:, 0:3*num_filters, 0:28, 0:28],
    #                     inputs2[:, 3*num_filters:, :, :])
    #
    #         # inputs2 = out1.detach().clone().cuda(1)
    #         start = time.time()
    #         # inputs2.requires_grad = True
    #         # inputs2.retain_grad()
    #         out2 = model2(inputs2_)
    #         # print("out2", type(out2), len(out2))
    #         # out_0, out_1 = out2[0].detach().clone().cuda(2), out2[1].detach().clone().cuda(2)
    #         # out_0.requires_grad = True
    #         # out_0.retain_grad()
    #         # out_1.requires_grad = True
    #         # out_1.retain_grad()
    #         # print("2", out_0.shape, out_1.shape)
    #         # inputs3 = (out_0, out_1)
    #         out_0, out_1 = out2
    #         # out_0 = torch.cat((out_0, out_0), dim=2)
    #         # out_0 = torch.cat((out_0, out_0), dim=3)
    #         # print("out0", out_0.shape)
    #         out2 = torch.cat((out_0, out_1), dim=0)
    #         time2 += time.time() - start
    #         inputs3 = out2.detach().clone().cuda(2)
    #         inputs3.requires_grad = True
    #         inputs3.retain_grad()
    #         # out_0.requires_grad = True
    #         # out_0.retain_grad()
    #         # out_1.requires_grad = True
    #         # out_1.retain_grad()
    #         # print("1", out_0.shape, out_1.shape)
    #         inputs3_ = (inputs3[0:batch_size, :, :, :],
    #                     inputs3[batch_size:, :, :, :])
    #
    #         # inputs3 = out2.detach().clone().cuda(2)
    #         start = time.time()
    #         # inputs3.requires_grad = True
    #         # inputs3.retain_grad()
    #         out3 = model3(inputs3_)
    #         # print("out3", type(out3), len(out3))
    #         # out_0, out_1 = out3[0].detach().clone().cuda(3), out3[1].detach().clone().cuda(3)
    #         # out_0.requires_grad = True
    #         # out_0.retain_grad()
    #         # out_1.requires_grad = True
    #         # out_1.retain_grad()
    #         # print("3", out_0.shape, out_1.shape)
    #         # inputs4 = (out_0, out_1)
    #         out_0, out_1 = out3
    #         out_0 = torch.cat((out_0, out_0), dim=2)
    #         out_0 = torch.cat((out_0, out_0), dim=3)
    #         # print("out0", out_0.shape)
    #         out3 = torch.cat((out_0, out_1), dim=1)
    #         time3 += time.time() - start
    #         inputs4 = out3.detach().clone().cuda(3)
    #         inputs4.requires_grad = True
    #         inputs4.retain_grad()
    #         # out_0.requires_grad = True
    #         # out_0.retain_grad()
    #         # out_1.requires_grad = True
    #         # out_1.retain_grad()
    #         # print("1", out_0.shape, out_1.shape)
    #         inputs4_ = (inputs4[:, 0:12*num_filters, 0:7, 0:7],
    #                     inputs4[:, 12*num_filters:, :, :])
    #
    #         # inputs4 = out3.detach().clone().cuda(3)
    #         start = time.time()
    #         # inputs4.requires_grad = True
    #         # inputs4.retain_grad()
    #         output = model4(inputs4_)
    #
    #         loss = F.cross_entropy(output, target)
    #         loss.backward()
    #         optimizer4.step()
    #         optimizer4.zero_grad()
    #         time4 += time.time() - start
    #
    #         # grad_up_0 = inputs4[0].grad.detach().clone().cuda(2)
    #         # grad_up_1 = inputs4[1].grad.detach().clone().cuda(2)
    #         grad_up = inputs4.grad.detach().clone().cuda(2)
    #         start = time.time()
    #         # out3.backward(grad_up)
    #         # out3[0].backward(grad_up_0, retain_graph=True)
    #         # out3[1].backward(grad_up_1)
    #         out3.backward(grad_up)
    #         optimizer3.step()
    #         optimizer3.zero_grad()
    #         time3 += time.time() - start
    #
    #         # grad_up_0 = inputs3[0].grad.detach().clone().cuda(1)
    #         # grad_up_1 = inputs3[1].grad.detach().clone().cuda(1)
    #         grad_up = inputs3.grad.detach().clone().cuda(1)
    #         start = time.time()
    #         out2.backward(grad_up)
    #         # out2[0].backward(grad_up_0, retain_graph=True)
    #         # out2[1].backward(grad_up_1)
    #         # out2.backward(grad_up)
    #         optimizer2.step()
    #         optimizer2.zero_grad()
    #         time2 += time.time() - start
    #
    #         # grad_up_0 = inputs2[0].grad.detach().clone().cuda(0)
    #         # grad_up_1 = inputs2[1].grad.detach().clone().cuda(0)
    #         grad_up = inputs2.grad.detach().clone().cuda(0)
    #         start = time.time()
    #         # out1[0].backward(grad_up_0, retain_graph=True)
    #         # out1[1].backward(grad_up_1)
    #         out1.backward(grad_up)
    #         optimizer1.step()
    #         optimizer1.zero_grad()
    #         time1 += time.time() - start
    #
    #         # data_len += inputs.shape[0]
    #         losses += loss.item()
    #
    #         if i % 10 == 9:
    #             print("loss", losses/i, '-' , time1, time2, time3, time4)
    #             print("throughout", data_len/(time1+time2+time3+time4))
    #             # print("loss-throughout", losses/i, data_len/(time.time() - start))


if __name__ == '__main__':
    main()

import argparse
import os
import random
import shutil
import time
import warnings

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp

import torch
import torch.nn as nn
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--num-layers', default=18, type=int)
parser.add_argument('--num-filters', default=208, type=int)
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')


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
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        args.distributed = False
        main_worker(args.gpu, ngpus_per_node, args)
    else:
        args.distributed = True
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    torch.cuda.set_device(args.gpu)
    model = amoebanetd(num_classes=1000, num_layers=args.num_layers,
                       num_filters=args.num_filters).cuda()
    if args.distributed:
        rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    optimizer = SGD(model.parameters(), lr=0.1)
    print("finish model")
    # for name, params in model.named_parameters():
    #     print(name, ":", params.size())
    time.sleep(3)
    batch_size = args.batch_size
    dataset_size = 1000
    inputs = torch.rand(batch_size, 3, 224, 224).cuda()
    target = torch.randint(1000, (batch_size, )).cuda()
    epochs = args.epochs
    for epoch in range(epochs):
        start = time.time()
        data_len = 0
        losses = 0.0
        for i in range(dataset_size):
            output = model(inputs)
            loss = F.cross_entropy(output, target)
            loss.backward()

            data_len += inputs.shape[0]
            losses += loss.item()

            if i % 10 == 9:
                print("loss-throughout", losses/i, '-', data_len/(time.time() - start))
                data_len = 0
                start = time.time()
            optimizer.step()
            optimizer.zero_grad()

# def main():
#     args = parser.parse_args()
#     model = amoebanetd_pipeline(num_classes=1000, num_layers=args.num_layers,
#                                 num_filters=args.num_filters)
#     model1 = model.stage0().cuda(0)
#     model2 = model.stage1().cuda(1)
#     model3 = model.stage2().cuda(2)
#     model4 = model.stage3().cuda(3)
#     optimizer1 = SGD(model1.parameters(), lr=0.1)
#     optimizer2 = SGD(model2.parameters(), lr=0.1)
#     optimizer3 = SGD(model3.parameters(), lr=0.1)
#     optimizer4 = SGD(model4.parameters(), lr=0.1)
#     print("finish model")
#     # for name, params in model.named_parameters():
#     #     print(name, ":", params.size())
#     time.sleep(3)
#     batch_size = args.batch_size
#     dataset_size = 1000
#     inputs = torch.rand(batch_size, 3, 224, 224).cuda(0)
#     target = torch.randint(1000, (batch_size, )).cuda(3)
#     epochs = args.epochs
#     for epoch in range(epochs):
#         print("epoch")
#         time1 = 0.0
#         time2 = 0.0
#         time3 = 0.0
#         time4 = 0.0
#         data_len = 0
#         losses = 0.0
#         for i in range(dataset_size):
#             data_len += inputs.shape[0]
#             start = time.time()
#             out1 = model1(inputs)
#             # print("out1", type(out1), len(out1), type(out1[0]), type(out1[1]))
#             # print("out1", out1[0].size(), out1[1].size())
#             out_0, out_1 = out1[0].detach().clone().cuda(1), out1[1].detach().clone().cuda(1)
#             out_0.requires_grad = True
#             out_0.retain_grad()
#             out_1.requires_grad = True
#             out_1.retain_grad()
#             inputs2 = (out_0, out_1)
#             time1 += time.time() - start
#
#             # inputs2 = out1.detach().clone().cuda(1)
#             start = time.time()
#             # inputs2.requires_grad = True
#             # inputs2.retain_grad()
#             out2 = model2(inputs2)
#
#             out_0, out_1 = out2[0].detach().clone().cuda(2), out2[1].detach().clone().cuda(2)
#             out_0.requires_grad = True
#             out_0.retain_grad()
#             out_1.requires_grad = True
#             out_1.retain_grad()
#             inputs3 = (out_0, out_1)
#             time2 += time.time() - start
#
#             # inputs3 = out2.detach().clone().cuda(2)
#             start = time.time()
#             # inputs3.requires_grad = True
#             # inputs3.retain_grad()
#             out3 = model3(inputs3)
#             # print("out3", type(out3), len(out3))
#
#             out_0, out_1 = out3[0].detach().clone().cuda(3), out3[1].detach().clone().cuda(3)
#             out_0.requires_grad = True
#             out_0.retain_grad()
#             out_1.requires_grad = True
#             out_1.retain_grad()
#             inputs4 = (out_0, out_1)
#             time3 += time.time() - start
#
#             # inputs4 = out3.detach().clone().cuda(3)
#             start = time.time()
#             # inputs4.requires_grad = True
#             # inputs4.retain_grad()
#             output = model4(inputs4)
#
#             loss = F.cross_entropy(output, target)
#             loss.backward()
#             optimizer4.step()
#             optimizer4.zero_grad()
#             time4 += time.time() - start
#
#             start = time.time()
#             grad_up_0 = inputs4[0].grad.detach().clone().cuda(2)
#             grad_up_1 = inputs4[1].grad.detach().clone().cuda(2)
#             # out3.backward(grad_up)
#             out3[0].backward(grad_up_0, retain_graph=True)
#             out3[1].backward(grad_up_1)
#             optimizer3.step()
#             optimizer3.zero_grad()
#             time3 += time.time() - start
#
#             start = time.time()
#             grad_up_0 = inputs3[0].grad.detach().clone().cuda(1)
#             grad_up_1 = inputs3[1].grad.detach().clone().cuda(1)
#             out2[0].backward(grad_up_0, retain_graph=True)
#             out2[1].backward(grad_up_1)
#             # out2.backward(grad_up)
#             optimizer2.step()
#             optimizer2.zero_grad()
#             time2 += time.time() - start
#
#             start = time.time()
#             grad_up_0 = inputs2[0].grad.detach().clone().cuda(0)
#             grad_up_1 = inputs2[1].grad.detach().clone().cuda(0)
#             out1[0].backward(grad_up_0, retain_graph=True)
#             out1[1].backward(grad_up_1)
#             # out1.backward(grad_up)
#             optimizer1.step()
#             optimizer1.zero_grad()
#             time1 += time.time() - start
#
#             # data_len += inputs.shape[0]
#             losses += loss.item()
#
#             if i % 10 == 9:
#                 print("loss", losses/i, '-' , time1, time2, time3, time4)
#                 print("throughout", data_len/(time1+time2+time3+time4))
#                 # print("loss-throughout", losses/i, data_len/(time.time() - start))

# def main():
#     args = parser.parse_args()
#     model = amoebanetd(num_classes=1000, num_layers=args.num_layers,
#                        num_filters=args.num_filters).cuda()
#     balance = [3, 6, 8, 7]
#     model_list = split_module(model, balance, [])
#     # print("model_list", type(model_list))
#     # print("model1", type(model_list[0]))
#     model1 = model_list[0].cuda(0)
#     model2 = model_list[1].cuda(1)
#     model3 = model_list[2].cuda(2)
#     model4 = model_list[3].cuda(3)
#     optimizer1 = SGD(model1.parameters(), lr=0.1)
#     optimizer2 = SGD(model2.parameters(), lr=0.1)
#     optimizer3 = SGD(model3.parameters(), lr=0.1)
#     optimizer4 = SGD(model4.parameters(), lr=0.1)
#     print("finish model")
#     # for name, params in model.named_parameters():
#     #     print(name, ":", params.size())
#     time.sleep(3)
#     batch_size = args.batch_size
#     num_filters = args.num_filters
#     dataset_size = 1000
#     inputs = torch.rand(batch_size, 3, 224, 224).cuda(0)
#     target = torch.randint(1000, (batch_size, )).cuda(3)
#     epochs = args.epochs
#     for epoch in range(epochs):
#         print("epoch")
#         time1 = 0.0
#         time2 = 0.0
#         time3 = 0.0
#         time4 = 0.0
#         data_len = 0
#         losses = 0.0
#         for i in range(dataset_size):
#             data_len += inputs.shape[0]
#             start = time.time()
#             out1 = model1(inputs)
#             # print("out1", type(out1), len(out1), type(out1[0]), type(out1[1]))
#             # print("out1", out1[0].size(), out1[1].size())
#             # print("out1", type(out1), len(out1))
#             # out_0, out_1 = out1[0].detach().clone().cuda(1), out1[1].detach().clone().cuda(1)
#             out_0, out_1 = out1
#             out_0 = torch.cat((out_0, out_0), dim=2)
#             out_0 = torch.cat((out_0, out_0), dim=3)
#             # print("out0", out_0.shape)
#             out1 = torch.cat((out_0, out_1), dim=1)
#             time1 += time.time() - start
#             inputs2 = out1.detach().clone().cuda(1)
#             inputs2.requires_grad = True
#             inputs2.retain_grad()
#             # out_0.requires_grad = True
#             # out_0.retain_grad()
#             # out_1.requires_grad = True
#             # out_1.retain_grad()
#             # print("1", out_0.shape, out_1.shape)
#             inputs2_ = (inputs2[:, 0:3*num_filters, 0:28, 0:28],
#                         inputs2[:, 3*num_filters:, :, :])
#
#             # inputs2 = out1.detach().clone().cuda(1)
#             start = time.time()
#             # inputs2.requires_grad = True
#             # inputs2.retain_grad()
#             out2 = model2(inputs2_)
#             # print("out2", type(out2), len(out2))
#             # out_0, out_1 = out2[0].detach().clone().cuda(2), out2[1].detach().clone().cuda(2)
#             # out_0.requires_grad = True
#             # out_0.retain_grad()
#             # out_1.requires_grad = True
#             # out_1.retain_grad()
#             # print("2", out_0.shape, out_1.shape)
#             # inputs3 = (out_0, out_1)
#             out_0, out_1 = out2
#             # out_0 = torch.cat((out_0, out_0), dim=2)
#             # out_0 = torch.cat((out_0, out_0), dim=3)
#             # print("out0", out_0.shape)
#             out2 = torch.cat((out_0, out_1), dim=0)
#             time2 += time.time() - start
#             inputs3 = out2.detach().clone().cuda(2)
#             inputs3.requires_grad = True
#             inputs3.retain_grad()
#             # out_0.requires_grad = True
#             # out_0.retain_grad()
#             # out_1.requires_grad = True
#             # out_1.retain_grad()
#             # print("1", out_0.shape, out_1.shape)
#             inputs3_ = (inputs3[0:batch_size, :, :, :],
#                         inputs3[batch_size:, :, :, :])
#
#             # inputs3 = out2.detach().clone().cuda(2)
#             start = time.time()
#             # inputs3.requires_grad = True
#             # inputs3.retain_grad()
#             out3 = model3(inputs3_)
#             # print("out3", type(out3), len(out3))
#             # out_0, out_1 = out3[0].detach().clone().cuda(3), out3[1].detach().clone().cuda(3)
#             # out_0.requires_grad = True
#             # out_0.retain_grad()
#             # out_1.requires_grad = True
#             # out_1.retain_grad()
#             # print("3", out_0.shape, out_1.shape)
#             # inputs4 = (out_0, out_1)
#             out_0, out_1 = out3
#             out_0 = torch.cat((out_0, out_0), dim=2)
#             out_0 = torch.cat((out_0, out_0), dim=3)
#             # print("out0", out_0.shape)
#             out3 = torch.cat((out_0, out_1), dim=1)
#             time3 += time.time() - start
#             inputs4 = out3.detach().clone().cuda(3)
#             inputs4.requires_grad = True
#             inputs4.retain_grad()
#             # out_0.requires_grad = True
#             # out_0.retain_grad()
#             # out_1.requires_grad = True
#             # out_1.retain_grad()
#             # print("1", out_0.shape, out_1.shape)
#             inputs4_ = (inputs4[:, 0:12*num_filters, 0:7, 0:7],
#                         inputs4[:, 12*num_filters:, :, :])
#
#             # inputs4 = out3.detach().clone().cuda(3)
#             start = time.time()
#             # inputs4.requires_grad = True
#             # inputs4.retain_grad()
#             output = model4(inputs4_)
#
#             loss = F.cross_entropy(output, target)
#             loss.backward()
#             optimizer4.step()
#             optimizer4.zero_grad()
#             time4 += time.time() - start
#
#             # grad_up_0 = inputs4[0].grad.detach().clone().cuda(2)
#             # grad_up_1 = inputs4[1].grad.detach().clone().cuda(2)
#             grad_up = inputs4.grad.detach().clone().cuda(2)
#             start = time.time()
#             # out3.backward(grad_up)
#             # out3[0].backward(grad_up_0, retain_graph=True)
#             # out3[1].backward(grad_up_1)
#             out3.backward(grad_up)
#             optimizer3.step()
#             optimizer3.zero_grad()
#             time3 += time.time() - start
#
#             # grad_up_0 = inputs3[0].grad.detach().clone().cuda(1)
#             # grad_up_1 = inputs3[1].grad.detach().clone().cuda(1)
#             grad_up = inputs3.grad.detach().clone().cuda(1)
#             start = time.time()
#             out2.backward(grad_up)
#             # out2[0].backward(grad_up_0, retain_graph=True)
#             # out2[1].backward(grad_up_1)
#             # out2.backward(grad_up)
#             optimizer2.step()
#             optimizer2.zero_grad()
#             time2 += time.time() - start
#
#             # grad_up_0 = inputs2[0].grad.detach().clone().cuda(0)
#             # grad_up_1 = inputs2[1].grad.detach().clone().cuda(0)
#             grad_up = inputs2.grad.detach().clone().cuda(0)
#             start = time.time()
#             # out1[0].backward(grad_up_0, retain_graph=True)
#             # out1[1].backward(grad_up_1)
#             out1.backward(grad_up)
#             optimizer1.step()
#             optimizer1.zero_grad()
#             time1 += time.time() - start
#
#             # data_len += inputs.shape[0]
#             losses += loss.item()
#
#             if i % 10 == 9:
#                 print("loss", losses/i, '-' , time1, time2, time3, time4)
#                 print("throughout", data_len/(time1+time2+time3+time4))
#                 # print("loss-throughout", losses/i, data_len/(time.time() - start))


if __name__ == '__main__':
    main()

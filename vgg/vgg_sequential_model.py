import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from torch.nn.parameter import Parameter
import itertools


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)


class Conv2d(nn.Module):
    def __init__(self, size1, size2, recompute=False):
        super(Conv2d, self).__init__()
        self.recompute = recompute
        self.layer = nn.Conv2d(size1, size2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        _initialize_weights(self.layer)

    def com_forward(self, inputs):
        re = self.relu(self.layer(inputs))
        # re = self.layer(inputs)
        return re

    def forward(self, inputs):
        if self.recompute:
            if not inputs.requires_grad:
                inputs = inputs + torch.zeros(1, dtype=inputs.dtype, device=inputs.device, requires_grad=True)
            re = checkpoint(self.com_forward, inputs)
        else:
            re = self.com_forward(inputs)
        # re = self.relu(re)
        return re


class Pool2d(nn.Module):
    def __init__(self, recompute=False):
        super(Pool2d, self).__init__()
        self.recompute = recompute
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def com_forward(self, inputs):
        re = self.layer(inputs)
        return re

    def forward(self, inputs):
        if self.recompute:
            re = checkpoint(self.com_forward, inputs)
        else:
            re = self.com_forward(inputs)
        return re


class Linear(nn.Module):
    def __init__(self, size1, size2, recompute=False, R=True):
        super(Linear, self).__init__()
        self.recompute = recompute
        self.layer = nn.Linear(size1, size2)
        self.relu = nn.ReLU(inplace=True)
        self.R = R
        _initialize_weights(self.layer)

    def com_forward(self, inputs):
        if self.R:
            re = self.relu(self.layer(inputs))
        else:
            re = self.layer(inputs)
        return re

    def forward(self, inputs):
        if self.recompute:
            re = checkpoint(self.com_forward, inputs)
        else:
            re = self.com_forward(inputs)
        return re


class Flatten(nn.Module):
    def forward(self, inputs):
        return torch.flatten(inputs, start_dim=1)
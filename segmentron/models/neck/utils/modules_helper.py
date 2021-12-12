import torch
import torch.nn as nn


def conv1X1(inp_ch, oup_ch, kernel_size=1, stride=1, padding=0,bias=False):
    return nn.Conv2d(inp_ch, oup_ch, kernel_size, stride, padding, bias=bias)


def conv3X3(inp_ch, oup_ch, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Conv2d(inp_ch, oup_ch, kernel_size, stride, padding, bias=bias)


def conv3X3_bnrelu(inp_ch, oup_ch, kernel_size=3, stride=1, bias=False):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp_ch, oup_ch, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(oup_ch),
        nn.ReLU(inplace=True)
    )

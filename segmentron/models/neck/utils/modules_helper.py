import torch
import torch.nn as nn


def conv3X3(inp_ch, oup_ch, kernel_size=3, stride=1):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp_ch, oup_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup_ch),
        nn.ReLU(inplace=True)
    )

import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=1):
    padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True)
    )
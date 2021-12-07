import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=1):
    padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


def ConvBNReLU6(in_ch, out_ch, kernel_size=3, stride=1):
    padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True)
    )
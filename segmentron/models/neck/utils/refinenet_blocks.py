import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules_helper import *



class ResidualConvUnit(nn.Module):
    def __init__(self, inp_ch):
        super(ResidualConvUnit, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inp_ch, inp_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(inp_ch, inp_ch, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        hx = self.relu(x)
        hx = self.conv1(hx)
        hx = self.relu(hx)
        hx = self.conv2(hx)

        out = x + hx
        return out

class MutiResolutionFusion(nn.Module):
    def __init__(self, oup_ch, *dim_of_features):
        super(MutiResolutionFusion, self).__init__()

        self.scale_factors = []
        _, max_size = max(dim_of_features, key=lambda x: x[1])

        for i, dim in enumerate(dim_of_features):
            inp_ch, inp_size = dim
            assert max_size % inp_size == 0, "max_size not divisble by dim {}".format(i)

            self.scale_factors.append(max_size // inp_size)
            self.add_module(
                f'conv{i}', nn.Conv2d(inp_ch, oup_ch, kernel_size=3, stride=1, padding=1, bias=False)
            )

    def forward(self, *xs):
        out = self.conv0(xs[0])
        if self.scale_factors[0] != 1:
            out = F.interpolate(
                out,
                scale_factor=self.scale_factors[0],
                mode='bilinear',
                align_corners=True)

        for i, x in enumerate(xs[1:], 1):
            x = getattr(self, f'conv{i}')(x)
            if self.scale_factors[i] != 1:
                x = F.interpolate(
                    x, 
                    scale_factor=self.scale_factors[i],
                    mode='bilinear',
                    align_corners=True)
            out += x

        return out

class ChainedResidualPool(nn.Module):
    def __init__(self, inp_ch):
        super(ChainedResidualPool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module(
                f'branch{i}', 
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(inp_ch, inp_ch, 3, 1, 1, bias=False)))
    
    def forward(self, x):
        out = self.relu(x)
        branch = out

        for i in range(1, 4):
            branch = getattr(self, f'branch{i}')(branch)
            out += branch

        return out


class RefinenetBlock(nn.Module):
    """RefineNet Block"""

    def __init__(self, oup_ch, *dim_of_features,
                 RCU=ResidualConvUnit,
                 MRF=MutiResolutionFusion,
                 CRP=ChainedResidualPool):
        """
        Args:
            oup_ch -> int: 
                代表最终每个block输出的特征图通道数
            dim_of_features -> tuple: (inp_ch, inp_size)
                代表输入的每个特征图的通道数和宽高尺寸
        """
        super(RefinenetBlock, self).__init__()

        num_features = len(dim_of_features)
        for i in range(num_features):
            in_channel, in_size = dim_of_features[i]

            double_rcu = [RCU(in_channel), RCU(in_channel)]
            self.add_module(f'rcu{i}', nn.Sequential(*double_rcu))
        
        if num_features > 1:
            self.mrf = MRF(oup_ch, *dim_of_features)
        else:
            self.mrf = None
        
        self.crp = CRP(oup_ch)
        self.conv = RCU(oup_ch)

    def forward(self, *xs):
        out = []
        for i, x in enumerate(xs):
            out.append(getattr(self, f'rcu{i}')(x))
        
        if self.mrf is not None:
            out = self.mrf(out)
        
        out = self.crp(out)
        out = self.conv(out)

        return out    
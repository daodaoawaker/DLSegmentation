import torch
import torch.nn as nn
import torch.nn.functional as F

# import os, sys
# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.sep.join(cur_path.split(os.sep)[:-3])
# sys.path.append(root_path)

from segmentron.models.utils import BACKBONE_REGISTRY
from segmentron.core.config import Cfg
from segmentron.models.backbone.utils import *



class Block(nn.Module):
    """Depthwise Conv + Pointwise Conv"""
    # 深度可分离卷积

    def __init__(self, in_ch, out_ch, stride=1):
        super(Block, self).__init__()
        # depthwise conv
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU6(inplace=True)
        # pointwise conv
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        return out

class MobileNetV1(nn.Module):
    """整个网络有28层，其中深度卷积层有13层"""

    # (128,2) means conv out_planes=128, conv stride=2, 
    # by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 
           512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, in_ch=3, num_class=1000, out_features_id=[7, 9, 11, 13]):
        super(MobileNetV1, self).__init__()
        self.out_features_id = out_features_id

        self.convin = ConvBNReLU(in_ch, 32, kernel_size=3, stride=2)
        self.layers = self._make_layers(in_planes=32)
        self.pool = nn.AvgPool2d(kernel_size=7)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(1024, num_class)

    def _make_layers(self, in_planes):
        layers = []
        for item in self.cfg:
            out_planes = item if isinstance(item, int) else item[0]
            stride = 1 if isinstance(item, int) else item[1]            
            layers.append(Block(in_planes, out_planes, stride=stride))
            in_planes = out_planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out_features = []
        hx = self.convin(x)

        for i, layer in enumerate(self.layers):
            hx = layer(hx)
            if i+1 in self.out_features_id:
                out_features.append(hx)
        
        hx = self.pool(hx)
        hx = hx.view(hx.size(0), -1)
        hx = self.fc(hx)

        return out_features



class InvertedResidual(nn.Module):
    """MobileNetV2中的linear bottleneck，作者称之为Inverted Residuals结构，同时使用到了resnet中的shortcut结构"""

    def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        mid_ch = in_ch * expand_ratio
        self.use_shortcut = self.stride == 1 and in_ch == out_ch

        self.conv = nn.Sequential(
            # PW conv
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            # DW conv
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            # PW linear-conv
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    """整个网络共54层"""

    bottleneck_cfg = [
        # t, c, n, s   ——> expand_ratio, out_channels, repeat_numbers, strides
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]

    def __init__(self, in_ch=3, num_class=1000, out_features_id=[]):
        super(MobileNetV2, self).__init__()
        self.out_features_id = out_features_id

        in_planes = 32
        out_planes = 1280

        self.convin = ConvBNReLU(in_ch, in_planes, kernel_size=3, stride=2)
        layers = []
        for t, c, n, s in self.bottleneck_cfg:
            layers.extend(self._make_layers(in_planes, t, c, n, s))
            in_planes = c
        self.layers = nn.Sequential(*layers)
        self.convout = ConvBNReLU(in_planes, out_planes, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=7)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(out_planes, num_class)

    def _make_layers(self, in_planes, t, c, n, s):
        outputs = []
        expand_ratio, out_planes, stride = t, c, s
        for i in range(n):
            stride = stride if i == 0 else 1
            outputs.append(InvertedResidual(in_planes, out_planes, expand_ratio, stride))
            in_planes = out_planes

        return outputs

    def forward(self, x):
        hx = self.convin(x)

        for i, layer in enumerate(self.layers):
            hx = layer(hx)
        
        hx = self.convout(hx)
        hx = self.pool(hx)
        hx = hx.view(hx.size(0), -1)
        hx = self.fc(hx)

        return hx



class MobileNetV3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MobileNetV3, self).__init__()

    def forward(self, x):
        pass




@BACKBONE_REGISTRY.register()
def mobilenetv1(*args, **kwargs):
    model = MobileNetV1(*args, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def mobilenetv2(*args, **kwargs):
    model = MobileNetV2(*args, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def mobilenetv3(*args, **kwargs):
    model = MobileNetV3(*args, **kwargs)
    return model



if __name__ == '__main__':
    x = torch.rand(5, 3, 224, 224)
    # model = MobileNetV1()
    model = MobileNetV2()
    
    out = model(x)
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, in_ch=3, num_class=1000, used_layers=[7, 9, 11, 13]):
        super(MobileNetV1, self).__init__()
        self.used_layers = used_layers

        features = []
        features.append(ConvBNReLU(in_ch, 32, kernel_size=3, stride=2))
        features.extend(self._make_layers(in_planes=32))
        self.features = features
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

        return layers

    def forward(self, x):
        outputs = []

        hx = x
        for i, layer in enumerate(self.features):
            hx = layer(hx)
            if i in self.used_layers:
                outputs.append(hx)
        hx = self.pool(hx)
        hx = hx.view(hx.size(0), -1)
        out = self.fc(hx)

        return outputs



class InvertedResidual(nn.Module):
    """MobileNetV2中的linear bottleneck，作者称之为Inverted Residuals结构，同时使用到了resnet中的shortcut结构"""

    def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], "stride must be 1 or 2."
        self.use_shortcut = stride == 1 and in_ch == out_ch
        mid_ch = in_ch * expand_ratio

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

    interverted_residual_cfg = [
        # t, c, n, s   ——> expand_ratio, out_channels, repeat_numbers, strides
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]

    def __init__(self, in_ch=3, num_class=1000, width_mult=1., used_layers=[3, 5, 7, 9]):
        super(MobileNetV2, self).__init__()
        self.used_layers = used_layers
        inp_channels = make_divisible(32 * width_mult, 8)
        oup_channels = make_divisible(1280 * max(1.0, width_mult), 8)

        features = []
        features.append(ConvBNReLU6(in_ch, inp_channels, 3, 2))
        inp_planes = inp_channels
        for t, c, n, s in self.interverted_residual_cfg:
            layers = []
            oup_planes = make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(inp_planes, oup_planes, t, stride))
                inp_planes = oup_planes
            features.append(nn.Sequential(*layers))
        features.append(ConvBNReLU6(inp_planes, oup_channels, 1, 1))

        self.features = nn.Sequential(*features)
        self.pool = nn.AvgPool2d(kernel_size=7)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(oup_channels, num_class)

    def forward(self, x):
        outputs = []

        hx = x
        for i, layer in enumerate(self.features, 1):
            hx = layer(hx)
            if i in self.used_layers:
                outputs.append(hx)
        hx = self.pool(hx)
        hx = hx.view(hx.size(0), -1)
        out = self.fc(hx)

        return outputs



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
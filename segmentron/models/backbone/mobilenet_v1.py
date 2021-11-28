import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentron.models.utils import ENCODER_REGISTRY, DECODER_REGISTRY
from segmentron.core.config import Cfg



class Block(nn.Module):
    """Depthwise Conv + Pointwise Conv"""

    def __init__(self, in_ch, out_ch, stride=1):
        super(Block, self).__init__()
        # depthwise conv
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_ch)
        self.relu1 = nn.ReLU6()
        # pointwise conv
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        return out


class MobileNetV1(nn.Module):
    """整个网络有28层，其中深度卷积层有13层"""

    # (128,2) means conv in_planes=128, conv stride=2, 
    # by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 
           512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, in_ch, num_class=1000):
        super(MobileNetV1, self).__init__()
        self.out_feature = [7, 9, 11, 13]

        self.convin = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.layers = self._make_layers(in_planes=32)
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.linear = nn.Linear(1024, num_class)

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
        hx = self.bn(hx)
        for i, layer in enumerate(self.layers):
            hx = layer(hx)
            if i+1 in self.out_feature:
                out_features.append(hx)
        out = self.pool(hx)
        out = self.linear(out)

        return out_features



@ENCODER_REGISTRY.register()
def mobilenetv1(*args, **kwargs):
    model = MobileNetV1(*args, **kwargs)
    return model
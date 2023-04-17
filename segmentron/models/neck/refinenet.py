import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentron.models.build import NECK_REGISTRY
from .utils.refinenet_blocks import RefinenetBlock, ResidualConvUnit
from .utils.modules_helper import *


class RefineNet(nn.Module):
    """Muti-path 4-Cascaded RefineNet for image segmentation

        网络的基本结构是RefineNet Block, 该block又主要由三个模块单元构成:
        1. RCU (Residual Convolution Unit)
        2. MRF (Multi Resolution Fusion)
        3. CRP (Chained Residual Pooling)
    """

    def __init__(self, input_size=224, mid_ch=256, num_class=1, used_layers=[0, 1, 2, 3]):
        super(RefineNet, self).__init__()

        self.layer1 = conv3X3(256, mid_ch)
        self.layer2 = conv3X3(512, mid_ch)
        self.layer3 = conv3X3(1024, mid_ch)
        self.layer4 = conv3X3(2048, mid_ch * 2)

        self.refinenet4 = RefinenetBlock(mid_ch * 2,
                                         (mid_ch * 2, input_size // 32))
        self.refinenet3 = RefinenetBlock(mid_ch,
                                         (mid_ch * 2, input_size // 32), (mid_ch, input_size // 16))
        self.refinenet2 = RefinenetBlock(mid_ch,
                                         (mid_ch, input_size // 16), (mid_ch, input_size // 8))
        self.refinenet1 = RefinenetBlock(mid_ch,
                                         (mid_ch, input_size // 8), (mid_ch, input_size // 4))

        self.out_conv = nn.Sequential(
            ResidualConvUnit(mid_ch),
            ResidualConvUnit(mid_ch),
            conv1X1(mid_ch, num_class, bias=True)
        )

    def forward(self, feature_list):

        x1, x2, x3, x4 = feature_list
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)
        x4 = self.layer4(x4)

        out4 = self.refinenet4(x4)
        out3 = self.refinenet3(out4, x3)
        out2 = self.refinenet2(out3, x2)
        out1 = self.refinenet1(out2, x1)

        output = [out4, out3, out2, out1]
        out = self.out_conv(out1)
    
        return output


@NECK_REGISTRY.register()
def refinenet(*args, **kwargs):
    model = RefineNet()
    
    return model
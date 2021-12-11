import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentron.models.utils import NECK_REGISTRY



class RefineNet(nn.Module):
    """Muti-path 4-Cascaded RefineNet for image segmentation

        网络的基本结构是RefineNet Block, 该block又主要由三个模块单元构成:
        1. RCU (Residual Convolution Unit)
        2. MRF (Multi Resolution Fusion)
        3. CRP (Chained Residual Pooling)
    """

    def __init__(self, mid_ch, num_class, ):
        super(RefineNet, self).__init__()

    def forward(self, feature_list):
        x1, x2, x3, x4 = feature_list
        



@NECK_REGISTRY.register()
def refinenet(*args, **kwargs):
    model = RefineNet()
    
    return model
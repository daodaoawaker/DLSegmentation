import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentron.models.utils import BACKBONE_REGISTRY
from segmentron.core.config import Cfg



class ShuffleNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ShuffleNet, self).__init__()

    

    def forward(self):
        pass



@BACKBONE_REGISTRY.register()
def shufflenet(*args, **kwargs):
    model = ShuffleNet(*args, **kwargs)
    return model
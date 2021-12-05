import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentron.models.utils import NECK_REGISTRY



class RefineNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RefineNet, self).__init__()

    def forward(self, x):
        pass



@NECK_REGISTRY.register()
def refinenet(*args, **kwargs):
    model = RefineNet()
    
    return model
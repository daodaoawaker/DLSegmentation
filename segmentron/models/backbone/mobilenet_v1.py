import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentron.models.utils import ENCODER_REGISTRY, DECODER_REGISTRY
from segmentron.core.config import Cfg



class MobileNetV1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MobileNetV1, self).__init__()

    

    def forward(self, x):
        pass





@ENCODER_REGISTRY.register()
@DECODER_REGISTRY.register()
def mobilenetv1(*args, **kwargs):
    model = MobileNetV1(*args, **kwargs)
    return model
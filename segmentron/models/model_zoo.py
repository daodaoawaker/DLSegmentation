from enum import Enum
import torch
import torch.nn as nn

from segmentron.models.utils import get_backbone, get_neck
from segmentron.core import Cfg



class ModelBuilder(nn.Module):
    def __init__(self,):
        super(ModelBuilder, self).__init__()

        self.encoder = self._get_backbone()
        self.decoder = self._get_neck()
        self.head = self._get_head()
    
    def _get_backbone(self):
        self.encoder_name = Cfg.MODEL.ENCODER.lower()
        self.in_ch = Cfg.MODEL.IN_CHANNEL
        self.num_class = Cfg.MODEL.NUM_CLASS

        if self.encoder_name:
            return get_backbone(self.encoder_name, in_ch=self.in_ch, num_class=self.num_class)
        return None
    
    def _get_neck(self):
        self.decoder_name = Cfg.MODEL.DECODER.lower()
        return get_neck(self.decoder_name)

    def _get_head(self):
        # self.head_name = Cfg.MODEL.HEAD.lower()
        # return get_head(self.head_name)
        return nn.Identity()

    def forward(self, x):
        if self.encoder:
            x = self.encoder(x)
            
        x = self.decoder(x)
        out = self.head(x)

        return out



def create_model():
        model = ModelBuilder()

        return model


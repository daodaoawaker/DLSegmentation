from enum import Enum

import torch
import torch.nn as nn

from segmentron.utils.registry import Registry
from segmentron.models.utils import get_backbone, get_neck
from segmentron.core.config import Cfg



# 模型注册表
MODEL_REGISTRY = Registry("MODEL")


class ModelBuilder(nn.Module):
    def __init__(self,):
        super(ModelBuilder, self).__init__()

        self.encoder = self.get_backbone()
        self.decoder = self.get_neck()
        self.head = self.get_head()
    
    def get_backbone(self):
        self.encoder_name = Cfg.MODEL.ENCODER.lower()
        self.in_ch = Cfg.TRAIN.IN_CHANNEL
        self.num_class = Cfg.TRAIN.NUM_CLASS

        if self.encoder_name:
            return get_backbone(self.encoder_name, in_ch=self.in_ch, num_class=self.num_class)
        return None
    
    def get_neck(self):
        self.decoder_name = Cfg.MODEL.DECODER.lower()
        return get_neck(self.decoder_name)

    def get_head(self):
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


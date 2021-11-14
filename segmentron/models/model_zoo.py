from enum import Enum

import torch
import torch.nn as nn

from segmentron.utils.registry import Registry
from segmentron.models.utils import get_encoder, get_decoder
from segmentron.core.config import Cfg


# 模型注册表
MODEL_REGISTRY = Registry("MODEL")


class ModelBuilder(nn.Module):
    def __init__(self,):
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.head = None
    
    def get_encoder(self):
        self.encoder_name = Cfg.MODEL.ENCODER.lower()
        if self.encoder_name:
            return get_encoder(self.encoder_name)
        return None
    
    def get_decoder(self):
        self.decoder_name = Cfg.MODEL.DECODER.lower()
        return get_decoder(self.decoder_name)

    def forward(self, x):
        if self.encoder:
            x = self.encoder(x)
        out = self.decoder(x)

        return out



def create_model():
        model = ModelBuilder()

        return model


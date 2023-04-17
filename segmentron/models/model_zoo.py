import torch
import torch.nn as nn

from segmentron.models.build import (
    get_backbone,
    get_neck,
    get_head
)
from segmentron.config import Cfg


class ModelBuilder(nn.Module):
    def __init__(self,):
        super(ModelBuilder, self).__init__()

        self.backbone = self._get_backbone()
        self.neck = self._get_neck()
        self.head = self._get_head()

    def _get_backbone(self):
        self.in_ch = Cfg.MODEL.IN_CHANNEL
        self.num_class = Cfg.MODEL.NUM_CLASSES
        self.backbone_name = Cfg.MODEL.BACKBONE.lower()
        if self.backbone_name:
            return get_backbone(self.backbone_name, in_ch=self.in_ch, num_class=self.num_class)

        return None

    def _get_neck(self):
        kwargs = Cfg.MODEL
        self.neck_name = Cfg.MODEL.NECK.lower()
        if self.neck_name:
            return get_neck(self.neck_name, **kwargs)

        return None

    def _get_head(self):
        self.head_name = Cfg.MODEL.HEAD.lower()
        if self.head_name:
            return get_head(self.head_name)

        return None

    def forward(self, x):
        if self.backbone:
            x = self.backbone(x)
        if self.neck:
            x = self.neck(x)
        if self.head:
            x = self.head(x)

        out = x
        return out



def create_model():
    '''Build model.'''
    return ModelBuilder()


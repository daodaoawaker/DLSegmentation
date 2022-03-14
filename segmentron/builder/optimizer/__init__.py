import torch
import torch.nn as nn

from segmentron.config import Cfg


def _get_parameters(model):
    pass


def get_optimizer(model):
    name = Cfg.OPTIMIZER.NAME.lower()

    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=Cfg.OPTIMIZER.LR)
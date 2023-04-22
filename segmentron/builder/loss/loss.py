import torch
import torch.nn as nn
import torch.nn.functional as F


class MSE(nn.Module):
    def __init__(self, name):
        super(MSE, self).__init__()
        self.name = name
    
    def forward(self, *inputs):
        preds, targets = tuple(inputs)
        return nn.MSELoss(preds, targets)


class CrossEntropy(nn.Module):
    def __init__(self, name):
        super(CrossEntropy, self).__init__()
        self.name = name
    
    def forward(self, *inputs):
        preds, targets = tuple(inputs)
        return nn.CrossEntropyLoss(preds, targets)


class DiceLoss(nn.Module):
    def __init__(self, name):
        super(DiceLoss, self).__init__()
        self.name = name

    def forward(self, *inputs):
        preds, targets = tuple(inputs)

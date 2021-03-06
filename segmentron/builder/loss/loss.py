import torch
import torch.nn as nn
import torch.nn.functional as F



class CrossEntropy(nn.Module):
    def __init__(self,):
        super(CrossEntropy, self).__init__()
    
    def forward(self, *inputs):
        preds, targets = tuple(inputs)
        return nn.CrossEntropyLoss(preds, targets)


class DiceLoss(nn.Module):
    def __init__(self, ):
        super(DiceLoss, self).__init__()

    def forward(self, *inputs):
        preds, targets = tuple(inputs)

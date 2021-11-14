import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RefineNet, self).__init__()

    

    def forward(self, x):
        pass




def refinenet(*args, **kwargs):
    model = RefineNet()
    return model
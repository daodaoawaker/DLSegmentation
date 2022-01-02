import os
import random
import numpy as np
import torch
from segmentron.utils import distributed as dist



def snake2pascal(string):
    """ convert Snake case to Pascal case"""
    return string.replace('_', ' ').title().replace(' ', '')
    
def make_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def seed_for_all_rng(seed=None):
    """
    Set seed for random number generater in torch, numpy and python

    """
    if not seed:
        pass

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)
    torch.set_rng_state(rng.get_state())
    torch.cuda.manual_seed_all(seed)


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
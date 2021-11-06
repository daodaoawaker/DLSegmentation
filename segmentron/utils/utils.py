import os
import random
import numpy as np
import torch




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
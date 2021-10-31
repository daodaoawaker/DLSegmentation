import torch
import os
import random
import numpy as np
from tensorboardX import SummaryWriter

from segmentron.utils.logger import Logger
from segmentron.utils.distributed import dist_init
from segmentron.utils.utils import makedir_not_exists
from segmentron.core.config import Cfg


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

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_setup(args):
    # make directory if not exist
    makedir_not_exists(Cfg)

    # initialize distribute related
    dist_init()

    # set seed for all random number generator
    seed = args.seed
    seed_for_all_rng(seed)



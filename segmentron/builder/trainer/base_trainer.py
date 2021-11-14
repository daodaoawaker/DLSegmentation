import os
from importlib import import_module


import torch
import torch.backends.cudnn as cudnn

from segmentron.core.config import Cfg
from segmentron.utils.utils import *
from segmentron.utils.distributed import dist_init
from segmentron.models.model_zoo import create_model


class BaseTrainer:
    """ 
    训练器的基类，一般将训练过程中比较通用的方法放置于此实现
    """
    def __init__(self, local_rank, args):
        args.local_rank = local_rank
        args.distributed = local_rank >= 0
        args.device = torch.device(f'cuda:{args.local_rank}')

        self.args = args
        self.num_gpus = args.nprocs
        self.local_rank = local_rank
        self.device = args.device
        self.default_setup()
    
    def default_setup(self):
        # make directory if not exist
        if self.local_rank == 0:
            for dir in [Cfg.log_dir, Cfg.copy_dir, Cfg.output_dir]:
                make_if_not_exists(dir)

        # set seed for all random number generator
        seed_for_all_rng(self.args.seed + self.local_rank)

        # initalize process group
        if self.args.distributed:
            cudnn.benchmark = Cfg.CUDNN.BENCHMARK
            cudnn.deterministic = Cfg.CUDNN.DETERMINISTIC
            cudnn.enabled = Cfg.CUDNN.ENABLED
            dist_init(self.args)

    def create_meta_arch(self):
        task_type = Cfg.TASK.TYPE.lower()
        package_name = f'{task_type}_meta_arch'
        package = import_module(f'segmentron.apps.{task_type}.{package_name}')
        self.meta_arch = getattr(package, snake2pascal(package_name))()
        self.model = create_model()
        

import os
import pprint
from importlib import import_module


import torch
import torch.backends.cudnn as cudnn

from segmentron.core.config import Cfg
from segmentron.utils.utils import *
from segmentron.utils.logger import Logger
from segmentron.utils.distributed import dist_init
from segmentron.builder.dataloader import DataloaderBuilder
from segmentron.builder.loss import get_loss
from segmentron.builder.optimizer import get_optimizer
from segmentron.builder.scheduler import get_lr_scheduler



class BaseTrainer:
    """ 
    训练器的基类，一般将训练过程中比较通用的方法放置于此实现
    """
    def __init__(self, local_rank, args):
        args.local_rank = local_rank
        args.distributed = local_rank >= 0

        self.args = args
        self.num_gpus = args.nprocs
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{args.local_rank}')
        self.logger = Logger.logger
        self.tb_writer = Logger.tbWriter

        # 配置信息
        self.default_setup()
        self.config_info()

        # data
        self.dataloader = DataloaderBuilder(args)
        self.train_dataloader = self.dataloader.train_dataloader()
        self.valid_dataloader = self.dataloader.valid_dataloader()
        self.calib_dataloader = self.dataloader.calib_dataloader()
        # network
        self.meta_arch = self.create_meta_arch()
        self.model = self.meta_arch.model
        # loss
        self.criterion = get_loss(self.model)
        # optimizer
        self.optimizer = get_optimizer(self.model)
        # lr_scheduler
        self.lr_scheduler = get_lr_scheduler(self.model)



    def default_setup(self):
        # make directory if not exist
        if self.local_rank == 0:
            for dir in [Cfg.log_dir, Cfg.copy_dir, Cfg.output_dir]:
                make_if_not_exists(dir)

        # set seed for all random numbe r generator
        seed_for_all_rng(self.args.seed + self.local_rank)
        # initalize process group
        if self.args.distributed:
            cudnn.benchmark = Cfg.CUDNN.BENCHMARK
            cudnn.deterministic = Cfg.CUDNN.DETERMINISTIC
            cudnn.enabled = Cfg.CUDNN.ENABLED
            dist_init(self.args)

    def config_info(self):
        if self.local_rank == 0:
            self.logger.info(f"Using {self.num_gpus} GPUs.")
            self.logger.info(pprint.pformat(self.args))
            self.logger.info(Cfg)

    def create_meta_arch(self):
        task_type = Cfg.TASK.TYPE.lower()
        module_name = f'{task_type}_meta_arch'
        package_module = import_module(f'segmentron.apps.{task_type}.{module_name}')
        meta_arch = getattr(package_module, snake2pascal(module_name))()

        return meta_arch
        

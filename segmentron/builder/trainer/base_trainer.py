import os
import pprint
import logging
from importlib import import_module

import torch
import torch.backends.cudnn as cudnn

from tools import snake2pascal
from segmentron.utils.utils import *
from segmentron.utils.logger import Recorder
from segmentron.utils.distributed import dist_init
from segmentron.data import DataloaderBuilder
from segmentron.builder.loss import get_loss
from segmentron.builder.metric import get_metric
from segmentron.builder.optimizer import get_optimizer
from segmentron.builder.scheduler import get_lr_scheduler
from segmentron.config import Cfg


logger = logging.getLogger(Cfg.LOGGER_NAME)

class BaseTrainer:
    r"""Base class for train pipeline.

    There are the implementation of the general methods and attributes related to train.

    """
    def __init__(self, local_rank, args):
        args.local_rank = local_rank
        args.distributed = local_rank >= 0

        self.args = args
        self.num_gpus = args.nprocs
        self.local_rank = local_rank

        self.recorder = Recorder(args)
        self.logger = self.recorder.logger

        self.setup()
        self.cfg_info()

        # Model
        self.meta_arch()
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.criterion = get_loss(self.model)
        # Data
        self.dataloader = DataloaderBuilder(self.args)
        self.train_dataloader = self.dataloader.train_dataloader()
        self.valid_dataloader = self.dataloader.valid_dataloader()
        self.cali_dataloader = self.dataloader.cali_dataloader()
        self.test_dataloader = self.dataloader.test_dataloader()
        # Optimizer | lr_scheduler
        self.optimizer = get_optimizer(self.model)
        self.scheduler = get_lr_scheduler(self.optimizer)
        # Resume

        # Distribution
        self.model_dist()
        # Metric
        self.metric = get_metric(self.args)
        self.mean_score = 0.0
        self.best_score = 0.0
        # Others
        self.iters = 0
        self.epochs = 0
        self.cur_epoch = 0
        self.cur_iters = 0
        self.last_epoch = 0
        self.end_epoch = Cfg.TRAIN.END_EPOCH
        self.iters_per_epoch = len(self.train_dataloader.dataset) // Cfg.TRAIN.BATCH_SIZE_PER_GPU // self.num_gpus

    def setup(self):
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

    def cfg_info(self):
        if self.local_rank == 0:
            self.logger.info(f"Using {self.num_gpus} GPUs.")
            self.logger.info(pprint.pformat(self.args))
            self.logger.info(Cfg)

    def meta_arch(self):
        task_type = self.args.task.lower()
        module_name = f'{task_type}_meta_arch'
        package = import_module(f'segmentron.apps.{task_type}.{module_name}')
        meta_arch = getattr(package, snake2pascal(module_name))()

        self._meta_arch = meta_arch
        self.model = meta_arch.model

    def model_dist(self):
        """model distribute"""
        model = self.model
        self.device = torch.device(f'cuda:{self.local_rank}')
        model = model.to(self.device)

        if self.args.distributed:
            # DDP
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
        else:
            # DP
            model = torch.nn.DataParallel(model, device_ids=list(range(self.num_gpus)))

        self.model = model

    def save_checkpoint(self):
        save_file = f'epoch_{self.cur_epoch}_{self.cur_iters % self.iters_per_epoch:05d}.pth'
        save_path = os.path.join(Cfg.output_dir, save_file)
        self.logger.info(f'==> saving checkpoint to {save_path}')
        # saving of every epoch
        torch.save({
            'epoch': self.cur_epoch,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict() if isinstance(self.model, torch.nn.DataParallel) \
                                else self.model.module.state_dict()
        })
        # saving of best pth
        if self.mean_score > self.best_score:
            self.best_score = self.mean_score
            copy_path = os.path.join(Cfg.copy_dir, 'best.pth')
            model_state_dict = self.model.state_dict() if isinstance(self.model, torch.nn.DataParallel) \
                                        else self.model.module.state_dict()
            torch.save(model_state_dict, copy_path)
        
        msg = f'Loss: {self.valid_loss}, MeanScore: {self.mean_score}, BestScore: {self.best_score}'
        self.logger.info(msg)

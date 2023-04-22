import os
import torch
import logging

from segmentron.builder import Builder
from segmentron.builder.metric import get_metric
from segmentron.builder.optimizer import get_optimizer
from segmentron.builder.scheduler import get_lr_scheduler
from segmentron.config import Cfg


logger = logging.getLogger(Cfg.LOGGER_NAME)

class BaseTrainer(Builder):
    r"""Base class for train pipeline.

    There are the implementation of the general methods and attributes related to train.

    """
    def __init__(self, args):
        super(BaseTrainer, self).__init__(args)

        # Model
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.model = self.meta_arch.model

        # Data
        self.train_dataloader = self.dataloader.train_dataloader()
        self.valid_dataloader = self.dataloader.valid_dataloader()
        self.calib_dataloader = self.dataloader.calib_dataloader()

        # Optimizer | lr_scheduler
        self.optimizer = get_optimizer(self.model)
        self.scheduler = get_lr_scheduler(self.optimizer)

        # Resume


        # Distribution
        self.model_dist()

        # Metric
        self.mean_score = 0.0
        self.best_score = 0.0
        self.metric = get_metric(self.args)

        # Others
        self.iters = 0
        self.epochs = 0
        self.cur_epoch = 0
        self.cur_iters = 0
        self.last_epoch = 0
        self.end_epoch = Cfg.TRAIN.END_EPOCH
        self.iters_per_epoch = len(self.train_dataloader.dataset) // Cfg.TRAIN.BATCH_SIZE_PER_GPU // self.num_gpus

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

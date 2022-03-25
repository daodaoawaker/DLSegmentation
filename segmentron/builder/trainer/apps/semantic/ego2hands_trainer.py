import time
from importlib import import_module

from segmentron.builder.trainer import BaseTrainer, GeneralTrainer
from segmentron.utils.utils import *
from segmentron.utils import distributed as dist
from segmentron.config import Cfg



class Ego2HandsTrainer(GeneralTrainer):
    """
    Model Trainer
    
    """
    def __init__(self, local_rank, args):
        super(Ego2HandsTrainer, self).__init__(local_rank, args)

        if isinstance(self.criterion, (tuple, list)):
            assert len(self.criterion) == 2, "Ego2Hands use 2 loss for segmentations and detection."
            self.criterion_seg = self.criterion[0]
            self.criterion_det = self.criterion[1]

    def train(self,):
        self.logger.info(f"Begin to train, Total epochs: {self.epochs}, Total iters: {self.iters}")

        for epoch in range(self.last_epoch, self.end_epoch):
            # outside loop
            self.cur_epoch = epoch
            self.cur_iters = self.cur_epoch * self.iters_per_epoch
            if self.train_dataloader.sampler is not None and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            self._train()
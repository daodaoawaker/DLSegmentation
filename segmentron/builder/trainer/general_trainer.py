from importlib import import_module

from segmentron.builder.trainer import BaseTrainer
from segmentron.utils.utils import snake2pascal
from segmentron.utils import distributed as dist
from segmentron.core import Cfg



class GeneralTrainer(BaseTrainer):
    """
    Model Trainer
    
    """
    def __init__(self, local_rank, args):
        super(GeneralTrainer, self).__init__(local_rank, args)
        self.cfg = Cfg
        self.cur_epoch = self.last_epoch
        self.cur_iter = self.cur_epoch * self.iters_per_epoch

    def train(self,):
        for epoch in range(self.last_epoch, self.end_epoch):
            # outside loop
            self.cur_epoch = epoch
            if self.train_dataloader.sampler is not None and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            self._train()
            self._valid()
    
    def _train(self,):
        self.model.train()
        for i_iter, batch in enumerate(self.train_dataloader):
            self.cur_iter += i_iter
            images, labels = batch['image'], batch['label']
            images = images.cuda()
            labels = labels.long().cuda()

            preds = self.model(images)
            loss = self.criterion(preds, labels)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()


    def _valid(self,):
        pass


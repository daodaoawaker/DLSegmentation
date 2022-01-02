import time
from importlib import import_module

from segmentron.builder.trainer import BaseTrainer
from segmentron.utils.utils import *
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
        self.epochs = self.end_epoch - self.last_epoch
        self.iters = self.epochs * self.iters_per_epoch

    def train(self,):
        self.logger.info(f"Begin to train, Total epochs: {self.epochs}, Total iters: {self.iters}")

        for epoch in range(self.last_epoch, self.end_epoch):
            # outside loop
            self.cur_epoch = epoch
            self.cur_iters = self.cur_epoch * self.iters_per_epoch
            if self.train_dataloader.sampler is not None and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            self._train()
    
    def _train(self,):
        self.model.train()
        avg_loss = AverageMeter()
        batch_time = AverageMeter()
        tic = time.time()

        for i_iter, batch in enumerate(self.train_dataloader):
            self.cur_iters += 1

            images, labels = batch['image'], batch['label']
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(images)
            losses = self.criterion(preds, labels)
            loss = losses.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # update batch_time and average loss
            batch_time.update(time.time() - tic)
            tic = time().time()
            reduced_loss = reduce_tensor(loss) if dist.is_initialized() else loss
            avg_loss.update(reduced_loss.item())
            self.train_loss = avg_loss.average()

            # add item to tensorboard
            if self.cur_iters % Cfg.LOG_FREQ and self.local_rank == 0:
                self.recorder.scalar_summary('train_loss', self.train_loss, self.cur_iters)

            # print info of every iteration
            if i_iter % Cfg.PRINT_FREQ == 0 and self.local_rank == 0:
                msg = f'Epoch: [{self.cur_epoch}/{self.end_epoch}], Iters: [{i_iter}/{self.iters_per_epoch}], ' \
                    f'Lr: {self.scheduler.get_last_lr()}, Loss: {self.train_loss:.6f}, Time: {batch_time.average():.2f}'
                self.logger.info(msg)
            
        # validate after every epoch
        self._valid()
        self.model.train()
        
        if self.local_rank == 0:
            self._save_checkpoint()

    def _valid(self,):
        self.model.eval()
        avg_loss = AverageMeter()

        with torch.no_grad():
            for i_iter, batch in enumerate(self.valid_dataloader):
                images, labels = batch['image'], batch['label']
                images = images.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images)
                losses = self.criterion(preds, labels)

                loss = losses.mean()
                reduced_loss = reduce_tensor(loss) if dist.is_initialized() else loss
                avg_loss.update(reduced_loss.item())
                self.valid_loss = avg_loss.average()

                # TODO
                score = self.metric()
        
        self.mean_score = self.metric.get()
        if self.local_rank == 0:
            self.recorder.scalar_summary('valid_loss', self.valid_loss, self.cur_iters)

                



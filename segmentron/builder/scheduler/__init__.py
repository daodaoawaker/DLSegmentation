import torch

from segmentron.config import Cfg



def get_lr_scheduler(optimizer):
    lr_scheduler = None
    name = Cfg.TRAIN.LR_SCHEDULER.NAME.lower()

    if name == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            Cfg.TRAIN.LR_SCHEDULER.STEP_SIZE)
    elif name == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=Cfg.TRAIN.LR_SCHEDULER.GAMMA)

    return lr_scheduler
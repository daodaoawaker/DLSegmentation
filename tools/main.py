import _init_path
import torch
from importlib import import_module
import torch.multiprocessing as mp

from segmentron.utils.utils import *
from segmentron.config import Opt, Cfg



def train(proc_id, args):
    trainer_name = Cfg.TRAIN.TRAINER
    if 'general' in trainer_name:
        package = import_module(f'segmentron.builder.trainer.{trainer_name}')
    else:
        package = import_module(f'segmentron.builder.trainer.apps.{Cfg.TASK.TYPE}.{trainer_name}')
    
    trainer = getattr(package, snake2pascal(trainer_name))(proc_id, args)
    trainer.train()



if __name__ == "__main__":

    # load cfg
    args = Opt.args
    args.nprocs = torch.cuda.device_count()

    mp.spawn(train, nprocs=args.nprocs, args=[args])



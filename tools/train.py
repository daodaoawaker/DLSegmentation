import _init_path
import torch
from importlib import import_module
import torch.multiprocessing as mp

from segmentron.utils.utils import *
from segmentron.core import Opt, Cfg



def train(proc_id, nprocs, args):
    trainer_name = Cfg.TRAIN.TRAINER
    package = import_module('segmentron.builder.trainer.' + trainer_name)
    trainer = getattr(package, snake2pascal(trainer_name))(proc_id, args)
    trainer.train()


def main():
    # load cfg
    args = Opt.args

    args.nprocs = torch.cuda.device_count()
    mp.spawn(train, nprocs=args.nprocs, args=[args.nprocs, args])



if __name__ == "__main__":
    main()



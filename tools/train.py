import _init_path
import torch
from importlib import import_module
import torch.multiprocessing as mp

from segmentron.utils import ConfigParse
from segmentron.utils.utils import *
from segmentron.builder.trainer import *
from segmentron.core.config import Cfg   # 更新后的总配置



def train(proc_id, nprocs, args):
    trainer_name = Cfg.train.trainer
    package = import_module(trainer_name)
    trainer = getattr(package, snake2pascal(trainer_name))(proc_id, args)
    trainer.train()


def main():
    # load cfg
    opt = ConfigParse()
    args = opt.args

    args.nprocs = torch.cuda.deivce_count()
    mp.spawn(train, nprocs=args.nprocs, args=[args.nprocs, args])




if __name__ == "__main__":
    main()



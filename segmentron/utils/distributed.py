import torch

import torch.distributed as dist
from segmentron.core.config import Cfg


def dist_init(args):
    torch.cuda.set_device(args.device)
    dist.init_process_group(backend='nccl', init_method=Cfg.DIST.INIT_METHOD,
                            world_size=args.nprocs, rank=args.local_rank)
    # dist.init_process_group(backend='gloo', init_method=Cfg.DIST.INIT_METHOD,
    #                         world_size=args.nprocs, rank=args.local_rank)
    
    synchronize()


def synchronize():   # ???
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

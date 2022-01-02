import torch

import torch.distributed as torch_dist
from segmentron.core import Cfg



def dist_init(args):
    torch.cuda.set_device(args.local_rank)
    # torch_dist.init_process_group(backend='nccl', init_method=Cfg.DIST.INIT_METHOD,
    #                           world_size=args.nprocs, rank=args.local_rank)
    torch_dist.init_process_group(backend='gloo', init_method=Cfg.DIST.INIT_METHOD,
                                world_size=args.nprocs, rank=args.local_rank)
    synchronize()


def synchronize():   # ???
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not torch_dist.is_available():
        return
    if not torch_dist.is_initialized():
        return
    world_size = torch_dist.get_world_size()
    if world_size == 1:
        return
    torch_dist.barrier()


def is_initialized():
    return torch_dist.is_initialized()


def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()


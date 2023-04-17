import torch.multiprocessing as mp
from importlib import import_module
from tools import snake2pascal


def train_worker(proc_id, args):
    trainer_name = f'{args.trainer}_trainer'
    if 'general' in trainer_name:
        package = import_module(f'segmentron.builder.trainer.{trainer_name}')
    else:
        package = import_module(f'segmentron.builder.trainer.apps.{args.task}.{trainer_name}')

    trainer = getattr(package, snake2pascal(trainer_name))(proc_id, args)
    trainer.train()


def test_worker(proc_id, args):
    tester_name = f'{args.tester}_tester'
    if 'general' in tester_name:
        package = import_module(f'segmentron.builder.tester.{tester_name}')
    else:
        package = import_module(f'segmentron.builder.tester.apps.{args.task}.{tester_name}')

    tester = getattr(package, snake2pascal(tester_name))(proc_id, args)
    tester.test()


def train(args):
    """The entrance of Train pipeline"""

    mp.spawn(train_worker, nprocs=args.nprocs, args=[args])


def test(args):
    """The entrance of Test pipeline"""

    mp.spawn(test_worker, nprocs=args.nprocs, args=[args])


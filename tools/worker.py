import torch.multiprocessing as mp
from importlib import import_module
from tools import snake2pascal


def train_worker(proc_id, args):
    args.local_rank = proc_id
    trainer_module_name = f'{args.trainer}_trainer'
    import_module_path = \
        f'segmentron.builder.trainer.{trainer_module_name}' \
            if 'general' in trainer_module_name else \
                f'segmentron.builder.trainer.apps.{args.task}.{trainer_module_name}'
    package = import_module(import_module_path)
    trainer_class = snake2pascal(trainer_module_name)
    trainer = getattr(package, trainer_class)(args)
    trainer.train()


def test_worker(proc_id, args):
    args.local_rank = proc_id
    tester_module_name = f'{args.tester}_tester'
    import_module_path = \
        f'segmentron.builder.tester.{tester_module_name}' \
            if 'general' in tester_module_name else \
                f'segmentron.builder.tester.apps.{args.task}.{tester_module_name}'
    package = import_module(import_module_path)
    tester_class = snake2pascal(tester_module_name)
    tester = getattr(package, tester_class)(args)
    tester.test()


def train(args):
    """The entrance of Train pipeline."""
    mp.spawn(train_worker, nprocs=args.nprocs, args=[args])


def test(args):
    """The entrance of Test pipeline."""
    mp.spawn(test_worker, nprocs=args.nprocs, args=[args])


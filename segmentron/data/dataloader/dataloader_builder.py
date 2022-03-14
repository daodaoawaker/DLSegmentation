import torch
from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from segmentron.config import Cfg



class DataloaderBuilder:
    """数据加载器类"""

    def  __init__(self, args):
        self.cfg = Cfg
        self.args = args
        self.num_gpus = args.nprocs
        # only consider DP or DDP
        self.batch_size = Cfg.TRAIN.BATCH_SIZE_PER_GPU if self.args.distributed \
                                    else Cfg.TRAIN.BATCH_SIZE_PER_GPU * self.num_gpus

    def train_dataloader(self,):
        self.train_bs = self.batch_size
        self.train_dataset = self._get_dataset(mode='Train')
        self.train_sampler = self._get_sampler(mode='Train')
        self.train_loader = self._build_dataloader(mode='Train')

        return self.train_loader

    def valid_dataloader(self,):
        self.valid_bs = self.batch_size
        self.valid_dataset = self._get_dataset(mode='Valid')
        self.valid_sampler = self._get_sampler(mode='Valid')
        self.valid_loader = self._build_dataloader(mode='Valid')

        return self.valid_loader

    def test_dataloader(self,):
        pass

    def calib_dataloader(self,):
        pass

    def _get_dataset(self, mode='Train'):
        module_name = f'{Cfg.TASK.TYPE}'
        dataset_name = f'{Cfg.TASK.TYPE.title()}Dataset'
        augment_name = f'{Cfg.TASK.TYPE.title()}Augment'
        package_module = import_module('segmentron.apps.' + module_name)
        augment_class = getattr(package_module, augment_name)()  # ？？给什么参数
        dataset_class = getattr(package_module, dataset_name)(augment_class, mode)

        return dataset_class

    def _get_sampler(self, mode='Train'):
        sampler = None
        if self.args.distributed:
            if mode == 'Train':
                sampler = DistributedSampler(self.train_dataset)
            if mode == 'Valid':
                sampler = DistributedSampler(self.valid_dataset)

        return sampler

    def _build_dataloader(self, mode='Train'):
        dataloader = None
        if mode == 'Train':
            dataloader = DataLoader(self.train_dataset,
                                    batch_size=self.train_bs,
                                    shuffle=Cfg.TRAIN.SHUFFLE and self.train_sampler is None,
                                    num_workers=Cfg.WORKERS,
                                    pin_memory=True,
                                    sampler=self.train_sampler)
        if mode == 'Valid':
            dataloader = DataLoader(self.valid_dataset,
                                    batch_size=self.valid_bs,
                                    shuffle=False,
                                    num_workers=Cfg.WORKERS,
                                    pin_memory=True,
                                    sampler=self.valid_sampler)
        if mode == 'Test':
            pass
        if mode == 'Calib':
            pass

        return dataloader


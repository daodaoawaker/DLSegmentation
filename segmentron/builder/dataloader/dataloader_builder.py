import torch
from importlib import import_module
from torch.utils import data
from torch.utils.data import DataLoader, dataloader

from segmentron.core.config import Cfg
from segmentron.datasets



class DataloaderBuilder:
    """数据加载器类"""

    def  __init__(self, batch_size):
        self.cfg = Cfg
        self.batch_size = batch_size

    def train_dataloader(self,):
        train_dataset = self._get_dataset(mode='Train')
        train_loader = self._build_dataloader(train_dataset, mode='Train')

        return train_loader

    def valid_dataloader(self,):
        pass
    
    def test_dataloader(self,):
        pass

    def calib_dataloader(self,):
        pass

    def _get_dataset(self, mode='Train'):
        module_name = f'{Cfg.TASK.TYPE}_dataset'
        class_name = f'{Cfg.TASK.TYPE.tile()}Dataset'
        package_module = import_module('segmentron.apps.' + module_name)
        augment_class = self._get_transform(mode=mode)
        dataset_class = getattr(package_module, class_name)(augment_class, mode)

        return dataset_class

    def _get_transform(self, mode='Train'):
        augment_class = None
        return augment_class

    def _build_dataloader(self, dataset, mode='Train'):
        dataloader = DataLoader(dataset)

        return dataloader
    
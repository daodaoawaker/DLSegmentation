import torch
from importlib import import_module
from torch.utils import data
from torch.utils.data import DataLoader, dataloader

from segmentron.core.config import Cfg



class DataloaderBuilder:
    """数据加载器类"""

    def  __init__(self, batch_size):
        self.cfg = Cfg
        self.batch_size = batch_size

    def _get_dataset(self, mode='Train'):
        m_name = f'{Cfg.TASK.TYPE}_dataset'
        c_name = f'{Cfg.TASK.TYPE.tile()}Dataset'
        package_module = import_module('segmentron.apps.' + m_name)
        dataset = getattr(package_module, c_name)(mode)

        return dataset

    def _build_dataloader(self, dataset, mode='Train'):
        dataloader = DataLoader(dataset)

        return dataloader

    def train_dataloader(self,):
        train_dataset = self._get_dataset(mode='Train')
        train_loader = self._build_dataloader(train_dataset, mode='Train')

        return train_loader

    def valid_dataloader(self,):
        pass

    def calib_dataloader(self,):
        pass
    
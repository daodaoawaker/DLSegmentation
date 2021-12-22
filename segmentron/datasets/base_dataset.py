import torch
from torch.utils.data import Dataset

from segmentron.core.config import Cfg



class BaseDataset(Dataset):
    def __init__(self,
                 base_size=1024,
                 crop_size=(512, 512),
                 transform=None,
                 mode='Train',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        
        self.base_size = base_size
        self.crop_size = crop_size
        self.mode = mode
        self.mean = mean
        self.std = std

        self.datasets = []
        self.sample_path = []
        self.sample_list = []

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        raise NotImplementedError

    def read_samples(self,):
        raise NotImplementedError
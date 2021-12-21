import torch
from torch.utils.data import Dataset

from segmentron.core.config import Cfg



class BaseDataset(Dataset):
    def __init__(self, 
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.mean = mean
        self.std = std

        self.sample_list = []

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        raise NotImplementedError

    def read_samples(self,):
        raise NotImplementedError

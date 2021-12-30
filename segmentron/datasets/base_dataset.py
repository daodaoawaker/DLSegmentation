import torch
from torch.utils.data import Dataset

from segmentron.core import Cfg



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
        self.transform = transform
        self.mode = mode
        self.mean = mean
        self.std = std

        self.datasets = []       # 存放各个数据集的目录路径
        self.sample_path = []    # 存放所有的数据对的全路径 [(image1_path, label1_path), (image2_path, label2_path),  ... ]
        self.sample_list = []    # 存放所有的样本对应的字典 [{'image_path': path1, 'label_path': _path1}, { },  ... ]

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        raise NotImplementedError

    def read_samples(self,):
        raise NotImplementedError
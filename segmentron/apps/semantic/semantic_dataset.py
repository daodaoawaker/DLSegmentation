import os
import torch

from segmentron.datasets import BaseDataset
from segmentron.datasets.utils import get_sample_pair
from segmentron.core.config import Cfg



class SemanticDataset(BaseDataset):
    """"""

    def __init__(self, ):
        super(SemanticDataset, self).__init__(mean=Cfg.INPUT.MEAN, std=Cfg.INPUT.STD)
        self.cfg = Cfg

        self.sample_list = self.read_samples()

    def __getitem__(self, index):
        item = self.sample_list[index]



    def read_samples(self):
        samples = []

        image_path, label_path = get_sample_pair()
        img_name = os.path.splitext(os.path.basename(label_path))[0]
        samples.append({
            'name': img_name,
            'image_path': image_path,
            'label_path': label_path
        })

        return samples

import os
import torch
import numpy as np

from segmentron.datasets import BaseDataset
from segmentron.datasets.utils import *
from segmentron.core.config import Cfg



class SemanticDataset(BaseDataset):
    """语义分割Dataset类"""

    def __init__(self, transform=None, mode='Train'):
        super(SemanticDataset, self).__init__(transform=transform, mode=mode, mean=Cfg.INPUT.MEAN, std=Cfg.INPUT.STD)
        self.cfg = eval(f'Cfg.DATASET.{mode.upper()}SET')
        self.root = Cfg.DATASET.ROOT_DIR

        for dataset in self.cfg.NAMES:
            self.datasets.append(os.path.join(self.root, dataset))
        
        self.sample_path = get_sample_pair(self.datasets)
        self.sample_list = self.read_samples()

    def __getitem__(self, index):
        item = self.sample_list[index]
        name = item['name']
        image_path = item['image_path']
        img_rgb = load_image(image_path)
        shape = img_rgb.shape

        if 'Test' == self.mode:
            image = self.input_transform(img_rgb)

            return image.copy(), np.array(shape), name
        
        label_path = item['label_path']
        label = load_label(label_path)

        image, label = self.generate_sample(img_rgb, label)
        sample = {
            'image': image.copy(), 
            'label': label.copy(),
            'shape': np.array(shape),
            'name': name
        }

        return sample

    def read_samples(self):
        samples = []
        if 'Test' == self.mode:
            for item in self.sample_path:
                image_path = item
                name = os.path.split(os.path.basename(image_path))[0]
                samples.append({
                    'name': name,
                    'image_path': image_path
                })
        else:
            for item in self.sample_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                samples.append({
                    'name': name,
                    'image_path': image_path,
                    'label_path': label_path
                })

        return samples

    def generate_sample(self, image, label):
        if self.transform is not None:
            image, label = self.transform(image, label)


class SemanticAugment:
    """语义数据增强类"""

    def __init__(self,):
        pass

    def __call__(self, image, label):

        return image, label
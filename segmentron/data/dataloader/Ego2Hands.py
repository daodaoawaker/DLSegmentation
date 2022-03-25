import os
import cv2
import random
import numpy as np

from segmentron.data import BaseDataset
from segmentron.data.utils import *
from segmentron.data.augment import *
from segmentron.config import Cfg



class Ego2HandsData(BaseDataset):
    def __init__(self, transform=None, mode='Train'):
        super(Ego2HandsData, self).__init__(transform=transform, mode=mode)
        self.root = Cfg.DATASET.ROOT_DIR
        self.cfg = eval(f'Cfg.DATASET.{mode.upper()}SET')

        for dataset in self.cfg.NAMES:
            self.datasets.append(os.path.join(self.root, Cfg.DATASET.NAME, dataset))

        self.sample_path = get_sample_path_ego2hands(self.datasets, mode)
        self.sample_list = self.read_samples()

    def __getitem__(self, index):
        input_size = [int(i) for i in Cfg.INPUT.IMAGE_SIZE]

        # letf hand
        id_l = random.randint(0, self.__len__() - 1)
        left_src = cv2.imread(self.sample_list[id_l], cv2.IMREAD_UNCHANGED)
        shape = left_src.shape

        left_img = left_src.astype(np.float32)
        left_img = ReScale(input_size, cv2.INTER_CUBIC)(left_img)
        left_img = Flip('Horizontal')(left_img)

        left_seg = left_img[:, :, -1] > 128
        left_energy = cv2.imread(self.sample_list[id_l], 0).astype(np.float32)
        left_energy = ReScale(input_size, cv2.INTER_NEAREST)(left_energy)
        left_energy = Flip('Horizontal')(left_energy)



        image, label = self.generate_sample(left_src, left_seg)
        sample = {
            'image': image.copy(),
            'label': label.copy(),
            'shape': np.array(shape),
        }

        return sample

    def read_samples(self):
        samples = []
        if 'Test' == self.mode:
            for item in self.sample_path:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path))[0]
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

        return image, label


class Ego2HandsAugment:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image, label):

        return image, label
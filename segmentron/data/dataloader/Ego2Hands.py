import os
import cv2
import random
import numpy as np

from segmentron.data import BaseDataset
from segmentron.data.utils import *
from segmentron.data.augment import *
from segmentron.config import Cfg
from .Ego2Hands_utils import *



LEFT_IDX = 1
RIGHT_IDX = 2

class Ego2HandsData(BaseDataset):
    def __init__(self, transform=None, mode='Train'):
        super(Ego2HandsData, self).__init__(transform=transform, mode=mode)
        self.root = Cfg.DATASET.ROOT_DIR
        self.cfg = eval(f'Cfg.DATASET.{mode.upper()}SET')

        for dataset in self.cfg.NAMES:
            self.datasets.append(os.path.join(self.root, Cfg.DATASET.NAME, dataset))

        self.sample_path = get_sample_path_ego2hands(self.datasets, mode)
        self.sample_list = self.read_samples()
        self.bg_list = read_bg_data(os.path.join(self.root, Cfg.DATASET.NAME, Cfg.DATASET.BACKGROUND))

        self.input_edge = Cfg.MODEL.INPUT_EDGE
        self.valid_hand_seg_threshold = 5000

    def __getitem__(self, index):
        input_size = [int(i) for i in Cfg.INPUT.IMAGE_SIZE]

        # --------------------------------------- letf hand
        id_l = random.randint(0, self.__len__() - 1)
        left_src = cv2.imread(self.sample_list[id_l]['image_path'], cv2.IMREAD_UNCHANGED)
        assert left_src != None, f"Error, image not found: {self.sample_list[id_l]['image_path']}"

        left_img = left_src.astype(np.float32)
        left_img = ReScale(input_size, cv2.INTER_AREA)(left_img)
        left_img = Flip('Horizontal')(left_img)
        left_img_orig = left_img.copy()

        left_seg = left_img[:, :, -1] > 128
        left_energy = cv2.imread(self.sample_list[id_l]['label_path'], 0)
        left_energy = ReScale(input_size, cv2.INTER_AREA)(left_energy)
        left_energy = Flip('Horizontal')(left_energy)
        left_energy = left_energy.astype(np.float32) / 255.

        # augmentation
        left_img, left_seg, left_energy = random_translation(left_img, left_seg, left_energy)
        left_img = random_brightness(left_img, left_seg)
        left_img = random_smoothness(left_img)

        # --------------------------------------- right hand
        id_r = random.randint(0, self.__len__() - 1)
        right_src = cv2.imread(self.sample_list[id_r]['image_path'], cv2.IMREAD_UNCHANGED)
        assert right_src != None, f"Error, image not found: {self.sample_list[id_r]['image_path']}"

        right_img = right_img.astype(np.float32)
        right_img = ReScale(input_size, cv2.INTER_AREA)(right_img)
        right_img_orig = right_img.copy()

        right_seg = right_img[:, :, -1] > 128
        right_energy = cv2.imread(self.sample_list[id_r]['label_path'], cv2.IMREAD_GRAYSCALE)
        right_energy = ReScale(input_size, cv2.INTER_AREA)(right_energy)
        right_energy = right_energy.astype(np.float32) / 255.

        # augmentation
        right_img, right_seg, right_energy = random_translation(right_img, right_seg, right_energy)
        right_img = random_brightness(right_img, right_seg)
        right_img = random_smoothness(right_img)

        # --------------------------------------- background images
        bg_img = None
        while (bg_img is None):
            id_b = random.randint(0, len(self.bg_list) - 1)
            bg_img = cv2.imread(self.bg_list[id_b], cv2.IMREAD_COLOR).astype(np.float32)
            bg_img = random_bg_augment(bg_img)
            bg_img = random_smoothness(bg_img)

        # --------------------------------------- merge hands
        merge_mode = random.randint(0, 9)
        if merge_mode < 8:
            if np.sum(left_energy) > np.sum(right_energy):
                merge_mode = 0  # left hand first
            else:
                merge_mode = 4  # right hand first

        if merge_mode < 4:
            # left hand top, right hand bottom
            img_real, bg_img_resized = merge_hands(left_img, right_img, bg_img)
            img_real_orig, _ = merge_hands(left_img_orig, right_img_orig, bg_img_resized, bg_resize=False)
            
            seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
            seg_real[right_seg] = RIGHT_IDX
            seg_real[left_seg] = LEFT_IDX
            # if right-hand's size is insufficient, drop it
            right_mask = seg_real == RIGHT_IDX
            if right_mask.sum() < self.valid_hand_seg_threshold:
                seg_real[right_mask] = 0
                right_energy.fill(0.0)
        elif merge_mode >= 4 and merge_mode < 8:
            # left hand bottom, right hand top
            img_real, bg_img_resized = merge_hands(right_img, left_img, bg_img)
            img_real_orig, _ = merge_hands(right_img_orig, left_img_orig, bg_img_resized, bg_resize=False)
            seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
            seg_real[left_seg] = LEFT_IDX
            seg_real[right_seg] = RIGHT_IDX
            # if left-hand's size is insufficient, drop it
            left_mask = seg_real == LEFT_IDX
            if left_mask.sum() < self.valid_hand_seg_threshold:
                seg_real[left_mask] = 0
                left_energy.fill(0.0)
        elif merge_mode == 8:
            # drop left hand, right hand only
            img_real, bg_img_resized = merge_hands(right_img, None, bg_img)
            img_real_orig, _ = merge_hands(right_img_orig, None, bg_img_resized, bg_resize=False)
            seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
            seg_real[right_seg] = RIGHT_IDX
            left_energy.fill(0.0)
        elif merge_hands == 9:
            # drop right hand, left hand only
            img_real, bg_img_resized = merge_hands(left_img, None, bg_img)
            img_real_orig, _ = merge_hands(left_img_orig, None, bg_img_resized, bg_resize=False)
            seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
            seg_real[left_seg] = LEFT_IDX
            right_energy.fill(0.0)

        size_div_2 = [int(s / 2) for s in input_size]
        size_div_4 = [int(s / 4) for s in input_size]
        seg_real_div2 = ReScale(size_div_2, cv2.INTER_AREA)(seg_real)
        seg_real_div4 = ReScale(size_div_4, cv2.INTER_AREA)(seg_real)

        left_energy_div2 = ReScale(size_div_2, cv2.INTER_AREA)(left_energy)
        left_energy_div4 = ReScale(size_div_4, cv2.INTER_AREA)(left_energy_div2)

        right_energy_div2 = ReScale(size_div_2, cv2.INTER_AREA)(right_energy)
        right_energy_div4 = ReScale(size_div_4, cv2.INTER_AREA)(right_energy_div2)

        bg_energy = 1.0 - np.maximum(left_energy, right_energy)
        bg_energy_div2 = 1.0 - np.maximum(left_energy_div2, right_energy_div2)
        bg_energy_div4 = 1.0 - np.maximum(left_energy_div4, right_energy_div4)

        energy_gt = np.stack([bg_energy, left_energy, right_energy], axis=0)
        energy_gt2 = np.stack([bg_energy_div2, left_energy_div2, right_energy_div2], 0)
        energy_gt4 = np.stack([bg_energy_div4, left_energy_div4, right_energy_div4], 0)

        # 为何要转为灰度图
        img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2GRAY)

        if self.input_edge:
            img_edge = cv2.Canny(img_real.astype(np.uint8), 25, 100).astype(np.float32)
            img_real = np.stack([img_real, img_edge], axis=-1)
        else:
            img_real = np.expand_dims(img_real, axis=-1)

        image, label = self.generate_sample(left_src, left_seg)
        sample = {
            'image': image.copy(),
            'label': label.copy(),
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

    def generate_sample(self, objs):
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label


class Ego2HandsAugment:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image, label):

        return image, label
				


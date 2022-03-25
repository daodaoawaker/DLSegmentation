import os
import numpy as np

from segmentron.config import Cfg



def load_image(image_path):
    pass

def load_label(label_path):
    pass

def get_sample_pair(dataset_path_list, mode):
    pass

def get_sample_path_ego2hands(dataset_path_list, mode):
    train_pairs = []
    valid_pairs = []

    img_path_list = []
    seg_gt_path_list = []
    energy_l_path_list = []
    energy_r_path_list = []

    if 'Train' in mode:
        for dataset_path in dataset_path_list:

            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.endswith('.png') and "energy" not in f and "vis" not in f:
                        image_path = os.path.join(root, f)
                        label_path = image_path.replace('.png', '_energy.png')
                        train_pairs.append((image_path, label_path))

        return train_pairs

    elif 'Valid' in mode:
        for dataset_path in dataset_path_list:

            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.endswith(".png") and os.path.splitext(f)[0][-1].isdigit():
                        image_path = os.path.join(root, f)
                        label_path = os.path.join(root, f.replace(".png", "_seg.png"))
                        # energy_l_path_list.append(os.path.join(root, f.replace(".png", "_e_l.png")))
                        # energy_r_path_list.append(os.path.join(root, f.replace(".png", "_e_r.png")))
                        valid_pairs.append((image_path, label_path))

        return valid_pairs
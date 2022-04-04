import cv2
import os
import random
import math
import numpy as np

from segmentron.config import Cfg



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

def read_bg_data(bg_path):
    suffix = ['.jpg', '.png', '.jpeg']

    bg_path_list = []
    for root, dirs, files in os.walk(bg_path):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext in suffix:
                bg_path_list.append(os.path.join(root, f))
    return bg_path_list

def random_bg_augment(img, bg_adapt=False, brightness_aug=True, flip_aug=False):
    if brightness_aug:
        if bg_adapt:
            brightness_mean = int(np.mean(img))
            brightness_val = random.randint(brightness_mean - 50, brightness_mean + 50)
            img = random_brightness(img, None, brightness_val=brightness_val)
        else:
            brightness_val = random.randint(35, 220)
            img = random_brightness(img, None, brightness_val=brightness_val)
    if flip_aug:
        do_flip = random.getrandbits(1)
        if do_flip:
            img = cv2.flip(img, 1)
    return img

def resize_bg(fg_shape, bg_img, bg_adapt):
    fg_h, fg_w = fg_shape[:2]
    if not bg_adapt:
        bg_h, bg_w = bg_img.shape[:2]

        if bg_h < fg_h or bg_w < fg_w:
            h_ratio = float(fg_h / bg_h)
            w_ratio = float(fg_w / bg_w)
            ratio = max(h_ratio, w_ratio)

            tar_w, tar_h = math.ceil(bg_w*ratio), math.ceil(bg_h*ratio)
            bg_img = cv2.resize(bg_img, (tar_w, tar_h))

        bg_h, bg_w = bg_img.shape[:2]

        h_offset_range = max(bg_h - fg_h, 0)
        w_offset_range = max(bg_w - fg_w, 0)
        h_offset = random.randint(0, h_offset_range)
        w_offset = random.randint(0, w_offset_range)
        bg_img = bg_img[h_offset:h_offset+fg_h, w_offset:w_offset+fg_w, :3]
    else:
        bg_img = cv2.resize(bg_img, (fg_w, fg_h))
    return bg_img

def add_alpha_border(img):
    assert img.shape[-1] > 3, "Image shuold has alpha channel."
    fg_mask = (img[:, :, -1] == 0).astype(np.uint8)
    fg_mask = cv2.dilate(fg_mask, np.ones((3, 3)))
    alpha_mask = fg_mask * 255
    alpha_mask = 255 - cv2.GaussianBlur(alpha_mask, (7, 7), 0)
    img[:, :, -1] = alpha_mask
    img_seg = alpha_mask > 200
    img_all_seg = alpha_mask > 0
    return img, img_seg, img_all_seg

def add_alpha_image_to_bg(alpha_img, bg_img):
    alpha = (alpha_img[:, :, -1] / 255.)[:, :, None]
    alpha_s = np.repeat(alpha, 3, axis=2)
    alpha_l = 1.0 - alpha_s
    combined_img = np.multiply(alpha_s, alpha_img[:, :, :3]) + \
                        np.multiply(alpha_l, bg_img)
    return combined_img

def merge_hands(top_hand_img, bot_hand_img, bg_img, bg_adapt=False, bg_resize=True):
    assert top_hand_img is not None, 'There should be at least one hand to be merged.'
    bg_img_resized = resize_bg(top_hand_img.shape, bg_img, bg_adapt) if bg_resize else bg_img

    if bot_hand_img is not None:
        top_hand_img, _, _ = add_alpha_border(top_hand_img)
        bot_hand_img, _, _ = add_alpha_border(bot_hand_img)
        final_hand_img = add_alpha_image_to_bg(bot_hand_img, bg_img_resized)
        final_hand_img = add_alpha_image_to_bg(top_hand_img, final_hand_img)
    else:
        top_hand_img, _, _ = add_alpha_border(top_hand_img)
        final_hand_img = add_alpha_image_to_bg(top_hand_img, bg_img_resized)
    return final_hand_img, bg_img_resized

def random_translation(img, seg, energy):
    img_h, img_w = img.shape[:2]
    fg_mask = seg.copy()
    coords1 = np.where(fg_mask)
    img_top, img_bot = np.min(coords1[0]), np.max(coords1[0])

    shift_range_ratio = 0.2
    
    # down shift
    down_shift = True if not fg_mask[0, :].any() else False
    if down_shift:
        down_space = int((img_h - img_top) * shift_range_ratio)
        down_offset = random.randint(0, down_space)

        old_top = 0
        old_bot = img_h
        old_bot -= down_offset

        cut_height = old_bot - old_top
        new_top = img_h - cut_height
        new_bot = img_h
    else:
        old_top, old_bot = 0, img_h
        new_top, new_bot = old_top, old_bot

    # left/right shift
    coords2 = np.where(fg_mask[old_top:old_bot, :])
    img_left, img_right = np.min(coords2[1]), np.max(coords2[1])

    left_shift = True if fg_mask[old_top:old_bot, 0].any() else False
    right_shift = True if fg_mask[old_top:old_bot, -1].any() else False
    if left_shift and right_shift:
        if random.random() > 0.5:
            left_shift = False
        else:
            right_shift = False

    if left_shift:
        left_space = int(img_right * shift_range_ratio)
        left_offset = random.randint(0, left_space)
        old_left = 0
        old_left += left_offset
        old_right = img_w

        cut_width = old_right - old_left
        new_left = 0
        new_right = new_left + cut_width

    if right_shift:
        right_space = int((img_w - img_left) * shift_range_ratio)
        right_offset = random.randint(0, right_space)
        old_left = 0
        old_right = img_w
        old_right += right_offset

        cut_width = old_right - old_left
        new_left = img_w - cut_width
        new_right = img_w

    if not (left_shift or right_shift):
        old_left, old_right = 0, img_w
        new_left, new_right = old_left, old_right
    
    img_ = np.zeros_like(img)
    seg_ = np.zeros_like(seg)
    energy_ = np.zeros_like(energy)

    img_[new_top:new_bot, new_left:new_right] = img[old_top:old_bot, old_left:old_right]
    seg_[new_top:new_bot, new_left:new_right] = seg[old_top:old_bot, old_left:old_right]
    energy_[new_top:new_bot, new_left:new_right] = energy[old_top:old_bot, old_left:old_right]

    return img_, seg_, energy_

def get_random_brightness(bg_adapt=False, custom=False, seq_i=-1):
    dark_lighting_set = [5]
    normal_lighting_set = [1, 3, 4, 6, 7]
    bright_lighting_set = [2, 8]
    brightness_map = {"dark": (0, 55), "normal": (55, 200), "bright": (55, 255)}
    if not bg_adapt:
        return random.randint(15, 240)
    else:
        if not custom:
            if seq_i in dark_lighting_set:
                return random.randint(*brightness_map["dark"])
            elif seq_i in normal_lighting_set:
                return random.randint(*brightness_map["normal"])
            elif seq_i in bright_lighting_set:
                return random.randint(*brightness_map["bright"])
        else:
            custom_scene_brightness = Cfg.AUGMENT.BRIGHTNESS_MODE
            assert custom_scene_brightness != "", "Error: custom scene brightness not set. Please set \"custom_scene_brightness\" in the config file."
            assert custom_scene_brightness in brightness_map, "Error: unrecognized brightness {} (valid options [\"dark\", \"normal\", \"bright\"]".format(config.custom_scene_brightness)
            return random.randint(*brightness_map[custom_scene_brightness])

def random_brightness(img, seg, magnitude=20, brightness_val=None):
    brightness_val = get_random_brightness() if brightness_val is None else brightness_val

    old_mean = np.mean(img[seg]) if seg is not None else np.mean(img)
    assert old_mean != 0, "Image's mean value cannot be empty."
    new_mean = brightness_val + random.uniform(-magnitude/2, magnitude/2)
    img *= (new_mean / old_mean)
    img = np.clip(img, 0, 255)
    return img

def random_smoothness(img, smooth_rate=0.3):
    smooth_rate_tick = smooth_rate / 5
    rand_val = random.random()
    if rand_val < smooth_rate:
        if rand_val < smooth_rate_tick:
            kernel_size = 3
        elif rand_val < smooth_rate_tick * 2:
            kernel_size = 5
        elif rand_val < smooth_rate_tick * 3:
            kernel_size = 7
        elif rand_val < smooth_rate_tick * 4:
            kernel_size = 9
        else:
            kernel_size = 11
        
        img[:, :, :3] = cv2.blur(img[:, :, :3], (kernel_size, kernel_size))
    return img



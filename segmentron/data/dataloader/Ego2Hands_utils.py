import cv2
import os
import random
import numpy as np



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



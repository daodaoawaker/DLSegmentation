import numpy as np
import cv2


class ReScale:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, objs):
        if isinstance(objs, (list, tuple)):
            return [self.impl(item) for item in objs]
        else:
            return self.impl(objs)

    def impl(self, obj):
        tar_h, tar_w = self.size[:2]
        retval = cv2.resize(obj, (tar_w, tar_h), interpolation=self.interpolation)
        return retval

class Flip:
    def __init__(self, flip_mode='H'):
        """
        flip_mode:
            Horizontal / Vertical / Horizontal&Vertical
        """
        self.flip_mode = flip_mode[0] if isinstance(flip_mode, str) else flip_mode

    def __call__(self, objs):
        if isinstance(objs, (list, tuple)):
            return [self.impl(item) for item in objs]
        return self.impl(objs)

    def impl(self, obj):
        if self.flip_mode == 1 or self.flip_mode.lower() == 'h':
            retval = cv2.flip(obj, 1)
        if self.flip_mode == 0 or self.flip_mode.lower() == 'v':
            retval = cv2.flip(obj, 0)
        if self.flip_mode == -1 or self.flip_mode.lower() == 'hv':
            retval = cv2.flip(obj, -1)
        return retval
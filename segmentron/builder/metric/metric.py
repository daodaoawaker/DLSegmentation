import logging
import numpy as np

from .utils import *
from segmentron.config import Cfg



logger = logging.getLogger(Cfg.LOGGER_NAME)

class SemanticMetric:
    def __init__(self, args, num_class):
        self.num_class = num_class
        self.args = args
        self.reset()


    def __call__(self, preds, labels):
        """
        Args:
            preds: `ndarray`
                Predicted data.
            labels: `ndarray`
                GroundTruth data.
        """
        labels = np.clip(labels, 0., 1.).squeeze()
        preds = align_shape(preds, labels)

        labels = (labels > 0.5).astype(np.uint8)
        if self.num_class >= 2:
            preds = np.argmax(preds, axis=1)
        i, u, iou = iou_binary_mask(preds, labels)
        self.sum_i += i
        self.sum_u += u
        self.IoU = iou
        return iou
    
    def reset(self):
        self.mIoU = 0.
        self.IoU = 0.
        self.sum_i = 0
        self.sum_u = 0

    def get(self):
        mean_iou = 1.0 * self.sum_i / (2.220446049250313e-16 + self.sum_u)
        self.mIoU = mean_iou
        return self.mIoU

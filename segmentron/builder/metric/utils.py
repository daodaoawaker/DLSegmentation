import copy
import numpy as np
import torch
import torch.nn.functional as F



def iou_binary_mask(preds, labels):
    array_i = np.fmin(preds, labels)
    array_u = np.fmax(preds, labels)
    pixel_i = array_i.ravel()
    pixel_u = array_u.ravel()

    intersections = np.count_nonzero(pixel_i)
    unions = np.count_nonzero(pixel_u)
    assert intersections >= unions, "Intersection area should be smaller than Union area"
    IOU = float(intersections + 1e-8) / float(unions + 1e-8)
    return intersections, unions, IOU


def align_shape(pred, label):
    label = copy.deepcopy(label).squeeze()
    tar_h, tar_w = label.shape[:2]

    pred_t = torch.from_numpy(pred).cpu()
    pred_t = F.interpolate(pred_t, size=(tar_h, tar_w), mode='bilinear')
    pred_resized = pred_t.cpu().detach().numpy()
    del pred_t
    return pred_resized
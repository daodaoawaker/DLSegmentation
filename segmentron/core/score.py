import torch
import numpy as np

from segmentron.core import Cfg


class SemanticMetric:
    def __init__(self, num_class, args):
        self.args = args
        self.num_class = num_class



def get_metric(args):
    metric_class_name = f'{Cfg.TASK.TYPE.title()}Metric'
    num_class = Cfg.MODEL.NUM_CLASS
    metric = eval(metric_class_name)(num_class, args)

    return metric
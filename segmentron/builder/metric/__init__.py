from segmentron.config import Cfg
from .metric import SemanticMetric



def get_metric(args):
    metric_class = f'{Cfg.TASK.TYPE.title()}Metric'
    num_class = Cfg.MODEL.NUM_CLASS
    metric = eval(metric_class)(args, num_class)
    return metric


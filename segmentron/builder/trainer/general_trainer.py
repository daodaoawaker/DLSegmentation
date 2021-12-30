from importlib import import_module

from segmentron.builder.trainer import BaseTrainer
from segmentron.core import Cfg
from segmentron.utils.utils import snake2pascal



class GeneralTrainer(BaseTrainer):
    """
    Model Trainer
    
    """
    def __init__(self, local_rank, args):
        super(GeneralTrainer, self).__init__(local_rank, args)
        self.cfg = Cfg

    def train(self,):
        pass

    def eval(self,):
        pass


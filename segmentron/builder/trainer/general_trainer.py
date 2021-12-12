
import pprint
from importlib import import_module

from segmentron.builder.trainer import BaseTrainer
from segmentron.utils.logger import Logger
from segmentron.core.config import Cfg
from segmentron.utils.utils import snake2pascal



class GeneralTrainer(BaseTrainer):
    """
    Model Trainer
    
    """
    def __init__(self, local_rank, args):
        super(GeneralTrainer, self).__init__(local_rank, args)
        self.cfg = Cfg
        
        self.logger = Logger.logger
        self.tb_writer = Logger.tbWriter
        self.config_info()

        # ---------------------------------- create meta architeture ----------------------------------
        self.meta_arch = self.create_meta_arch()
        self.model = self.meta_arch.model

        # create criterion
        self.loss = None
        # optimizer

        # lr_scheduler

    def config_info(self):
        if self.local_rank == 0:
            self.logger.info(f"Using {self.num_gpus} GPUs.")
            self.logger.info(pprint.pformat(self.args))
            self.logger.info(Cfg)

    def train(self,):
        pass

    def eval(self,):
        pass


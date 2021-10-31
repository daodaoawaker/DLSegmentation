


from segmentron.models.model_zoo import get_model
from segmentron.utils.logger import Logger
from segmentron.core.config import Cfg



class Trainer:
    """
    Model Trainer
    
    """
    def __init__(self, args):
        self.cfg = Cfg
        
        self.logger = Logger.logger
        self.tbWriter = Logger.tbWriter


        # create model
        self.model = get_model(self.cfg)

        # create criterion

        # optimizer

        # lr_scheduler


    def train(self,):
        pass

    def validation(self,):
        pass

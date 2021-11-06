

from segmentron.core.config import Cfg

class BaseMetaArch:
    def __init__(self, ):
        self.cfg = Cfg
    
    def preprocess(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError
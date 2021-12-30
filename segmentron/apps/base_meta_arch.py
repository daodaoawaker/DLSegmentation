

from segmentron.core import Cfg
from segmentron.models.model_zoo import create_model



class BaseMetaArch:
    def __init__(self, ):
        self.cfg = Cfg
        self.model = create_model()
    
    def preprocess(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def prepare_quant(self):
        # self.model = Quantization(self.model)
        pass
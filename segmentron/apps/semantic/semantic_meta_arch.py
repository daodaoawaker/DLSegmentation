


from segmentron.apps import BaseMetaArch
from segmentron.core.config import Cfg
from segmentron.models.model_zoo import create_model



class SemanticMetaArch(BaseMetaArch):
    def __init__(self):
        super(SemanticMetaArch, self).__init__()
        self.model = create_model()

    def preprocess(self):
        pass

    def predict(self):
        pass

    def postprocess(self):
        pass

    def loss(self):
        pass

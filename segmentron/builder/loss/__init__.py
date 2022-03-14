from segmentron.config import Cfg
from .loss import CrossEntropy



def get_loss(model, **kwargs):
     loss_name = Cfg.LOSS.NAME.lower()

     if loss_name == 'crossentropy':
          return CrossEntropy(**kwargs)

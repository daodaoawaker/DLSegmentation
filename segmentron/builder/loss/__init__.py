from enum import Enum, unique

from .loss import (
     MSE,
     DiceLoss,
     CrossEntropy,
)
from segmentron.config import Cfg


@unique
class LossKind(Enum):
     MSE              = 0
     CrossEntropy     = 1
     DiceLoss         = 2
     IouLoss          = 3
     FocalLoss        = 4

     def to_class(self, *args, **kwargs):
          return _LOSS_NAME_MAPPING[self](self._name_, *args, **kwargs)


_LOSS_NAME_MAPPING = {
     LossKind.MSE:             MSE,
     LossKind.CrossEntropy:    CrossEntropy,
     LossKind.DiceLoss:        DiceLoss,
}


def build_loss(model, *args, **kwargs):
     loss_name = Cfg.LOSS.TYPE

     for name, member in LossKind.__members__.items():
          if loss_name == name:
               return LossKind[loss_name].to_class(*args, **kwargs)
     raise KeyError(f'There is no `{loss_name}` loss type, please check the specified loss name')
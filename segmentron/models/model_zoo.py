from enum import Enum

from segmentron.utils.registry import Registry
from segmentron.core.config import Cfg


MODEL_REGISTRY = Registry("MODEL")

class ModelBuildMode(Enum):
    Custom   = 'Custom'
    General  = 'General'


def create_model():
    model_build_mode = Cfg.MODEL.MODE
    model_build_mode = ModelBuildMode.Custom

    if model_build_mode == "Custom":
        model = MODEL_REGISTRY.get(name='ModelBuilder')
    else:
        model_name = Cfg.MODEL.NAME
        model = MODEL_REGISTRY.get(name=model_name)()

    return model
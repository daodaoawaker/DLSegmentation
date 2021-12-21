from segmentron.utils.registry import Registry



BACKBONE_REGISTRY = Registry("BACKBONE")
NECK_REGISTRY = Registry("NECK")

def get_backbone(name, **kwargs):
    model = BACKBONE_REGISTRY.get(name)(**kwargs)

    return model


def get_neck(name, **kwargs):
    model = NECK_REGISTRY.get(name)(**kwargs)

    return model


def get_head(name, **kwargs):
    pass
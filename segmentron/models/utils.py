import torch

from segmentron.utils.registry import Registry



ENCODER_REGISTRY = Registry("ENCODER")
DECODER_REGISTRY = Registry("DECODER")


def get_encoder(name, **kwargs):
    model = ENCODER_REGISTRY.get(name)(**kwargs)

    return model


def get_decoder(name, **kwargs):
    model = DECODER_REGISTRY.get(name)(**kwargs)

    return model
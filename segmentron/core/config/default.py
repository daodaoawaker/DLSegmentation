from  yacs.config import CfgNode as CN


_C = CN()

# Basic config info
_C.LOGGER_NAME = 'DLSegment'
_C.TENSORBOARD_LOG_DIR = ''
_C.OUTPUT_DIR = ''



# Model related
_C.MODEL = CN()

# Train related
_C.TRAIN = CN()
_C.TRAIN.LOG_DIR = ''

# Augment related

# Eval related
_C.TEST = CN()


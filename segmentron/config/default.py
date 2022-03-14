from yacs.config import CfgNode as CN


_C = CN()

# -------------------------------------------- Basic info --------------------------------------------

_C.LOGGER_NAME    = 'DLSegment'
_C.WORKERS = 4
_C.LOG_FREQ = 300
_C.PRINT_FREQ = 20

_C.project        = ''              # 创建的工程目录名
_C.log_dir        = 'logs'
_C.copy_dir       = 'copy'
_C.output_dir     = 'output'

# -------------------------------------------- Cudnn related --------------------------------------------

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = True
_C.CUDNN.ENABLED = True

# -------------------------------------------- Task related --------------------------------------------

_C.TASK = CN()
_C.TASK.TYPE = 'semantic'

# -------------------------------------------- Model related --------------------------------------------

_C.MODEL = CN()
# model encoder. 其范围包含Backbone以及其他自定义的组合
_C.MODEL.ENCODER = ''    
# model decoder.   
_C.MODEL.DECODER = ''

_C.MODEL.IN_CHANNEL = 3
_C.MODEL.NUM_CLASS = 2

# -------------------------------------------- Dataset related --------------------------------------------

_C.DATASET = CN()
_C.DATASET.ROOT_DIR = ''
_C.DATASET.DATASET = ''

_C.DATASET.TRAINSET = CN()
_C.DATASET.TRAINSET.NAMES = ()

_C.DATASET.VALIDSET = CN()
_C.DATASET.VALIDSET.NAMES = ()

_C.DATASET.TESTSET = CN()
_C.DATASET.TESTSET.NAMES = ()

# ----- Augment -----
_C.DATASET.AUGMENT = CN()

# -------------------------------------------- Input related --------------------------------------------

_C.INPUT = CN()
_C.INPUT.SIZE = 224
_C.INPUT.MEAN = [0.0, 0.0, 0.0]
_C.INPUT.STD = [1.0, 1.0, 1.0]

# -------------------------------------------- Loss related --------------------------------------------

_C.LOSS = CN()
_C.LOSS.NAME = ''
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [1]
_C.LOSS.USE_OHEM = False

# -------------------------------------------- Train related --------------------------------------------

_C.TRAIN = CN()
_C.TRAIN.TRAINER = 'general_trainer'
_C.TRAIN.END_EPOCH = 300


_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = False

_C.TRAIN.OPTIMIZER = CN(new_allowed=True)
_C.TRAIN.OPTIMIZER.NAME = ''
_C.TRAIN.OPTIMIZER.LR = 1e-4

_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
_C.TRAIN.LR_SCHEDULER.NAME = ''

# -------------------------------------------- Valid related --------------------------------------------

_C.VALID = CN()

_C.VALID.BATCH_SIZE_PER_GPU = 16

# ------------------------------------------ Distributed related ------------------------------------------

_C.DIST = CN()
_C.DIST.INIT_METHOD = 'tcp://127.0.0.1:05032'

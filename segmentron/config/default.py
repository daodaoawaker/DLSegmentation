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

_C.TASK = CN(new_allowed=True)
_C.TASK.TYPE = 'semantic'

# -------------------------------------------- Model related --------------------------------------------

_C.MODEL = CN(new_allowed=True)
_C.MODEL.BACKBONE = ''    
_C.MODEL.NECK = ''
_C.MODEL.HEAD = ''

_C.MODEL.IN_CHANNEL = 3
_C.MODEL.NUM_CLASSES = 2

# -------------------------------------------- Dataset related --------------------------------------------

_C.DATASET = CN(new_allowed=True)
_C.DATASET.TYPE = 'general'
_C.DATASET.NAME = ''
_C.DATASET.ROOT_DIR = ''

_C.DATASET.TRAINSET = CN(new_allowed=True)
_C.DATASET.TRAINSET.NAMES = ()

_C.DATASET.VALIDSET = CN(new_allowed=True)
_C.DATASET.VALIDSET.NAMES = ()

_C.DATASET.TESTSET = CN(new_allowed=True)
_C.DATASET.TESTSET.NAMES = ()

# -------------------------------------------- Augment related --------------------------------------------
_C.AUGMENT = CN(new_allowed=True)



# -------------------------------------------- Input related --------------------------------------------

_C.INPUT = CN(new_allowed=True)
_C.INPUT.IMAGE_SIZE = [224, 224]
_C.INPUT.MEAN = [0.0, 0.0, 0.0]
_C.INPUT.STD = 1.0

# -------------------------------------------- Loss related --------------------------------------------

_C.LOSS = CN(new_allowed=True)
_C.LOSS.TYPE = ()
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [1]
_C.LOSS.USE_OHEM = False

# -------------------------------------------- Train related --------------------------------------------

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.TRAINER = 'general'
_C.TRAIN.END_EPOCH = 300


_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = False

_C.TRAIN.OPTIMIZER = CN(new_allowed=True)
_C.TRAIN.OPTIMIZER.NAME = ''
_C.TRAIN.OPTIMIZER.LR = 1e-4

_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
_C.TRAIN.LR_SCHEDULER.NAME = ''

# -------------------------------------------- Valid related --------------------------------------------

_C.VALID = CN(new_allowed=True)
_C.VALID.BATCH_SIZE_PER_GPU = 16

# --------------------------------------------- Test related --------------------------------------------

_C.TEST = CN(new_allowed=True)
_C.TEST.TESTER = 'general'
_C.TEST.END_EPOCH = 300


_C.TEST.BATCH_SIZE_PER_GPU = 16
_C.TEST.SHUFFLE = False

_C.TEST.OPTIMIZER = CN(new_allowed=True)
_C.TEST.OPTIMIZER.NAME = ''
_C.TEST.OPTIMIZER.LR = 1e-4

_C.TEST.LR_SCHEDULER = CN(new_allowed=True)
_C.TEST.LR_SCHEDULER.NAME = ''

# ------------------------------------------ Distributed related ------------------------------------------

_C.DIST = CN(new_allowed=True)
_C.DIST.INIT_METHOD = 'tcp://127.0.0.1:05032'
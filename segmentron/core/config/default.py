from  yacs.config import CfgNode as CN


_C = CN()

# -------------------------------------------- Basic info --------------------------------------------

_C.LOGGER_NAME    = 'DLSEG'

_C.project        = ''              # 创建的工程目录名
_C.log_dir        = 'logs'
_C.output_dir     = 'output'
_C.copy_dir       = 'copy'



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
_C.MODEL.MODE = 'General'                        # 若为Custom，则为支持自定义网络模式(backbone/neck/head)
_C.MODEL.NAME = ''           



# -------------------------------------------- Dataset related --------------------------------------------

_C.DATASET = CN()



# -------------------------------------------- Dataset augment --------------------------------------------

_C.AUGMENT = CN()



# -------------------------------------------- Train related --------------------------------------------

_C.TRAIN = CN()
_C.TRAIN.TRAINER = 'general_trainer'



# -------------------------------------------- Test related --------------------------------------------

_C.TEST = CN()



# -------------------------------------------- Distributed related --------------------------------------------

_C.DIST = CN()
_C.DIST.INIT_METHOD = 'tcp://127.0.0.1:05032'



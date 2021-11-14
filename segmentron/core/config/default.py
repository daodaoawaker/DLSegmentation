from  yacs.config import CfgNode as CN


_C = CN()

# -------------------------------------------- Basic info --------------------------------------------

_C.LOGGER_NAME    = 'DLSegment'

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
# model encoder. 其范围包含Backbone以及其他自定义的组合
_C.MODEL.ENCODER = ''    
# model decoder.   
_C.MODEL.DECODER = ''



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



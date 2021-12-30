import os
import sys
import logging
import time
from tensorboardX import SummaryWriter

from segmentron.core import Cfg



class Recorder:
    """
    分别记录程序运行时产生的日志信息和模型训练时需要记录在tensorboard中的一些中间变量信息

    """
    def __init__(self):
        # assert Cfg.LOGGER_NAME == "DLSegment", "Please make sure logger's name."
        self.logger = self.set_logger()
        self.tbWriter = SummaryWriter(Cfg.log_dir)


    def set_logger(self,):
        if not os.path.exists(Cfg.output_dir):
            os.makedirs(Cfg.output_dir)

        # 创建Logger对象
        logger = logging.getLogger(Cfg.LOGGER_NAME)

        # 分别为控制台和文件创建日志处理器
        timestamp = time.strftime('%Y-%m-%d  %H-%M-%S', time.localtime())
        file_name = os.path.join(Cfg.output_dir, f'{timestamp}.log')
        file_handler = logging.FileHandler(file_name, mode='w')
        console_handler = logging.StreamHandler(sys.stdout)

        # 设置输出日志的格式
        formatter = logging.Formatter("[%(asctime)s][%(name)s] %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 分别为Logger对象和Handler对象设置输出日志等级
        logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)

        # Logger对象可以设置多个Handler对象
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    
    def scalar_summary(self,):
        # TODO
        self.tbWriter.add_scalar()

    def scalar_list_summary(self, ):
        # TODO
        pass

    def image_summary(self,):
        # TODO
        self.tbWriter.add_image()



Logger = Recorder()

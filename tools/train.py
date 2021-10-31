import _init_path

from segmentron.utils import ConfigParse
from segmentron.utils.setup import default_setup
from segmentron.core import Trainer
from segmentron.core.config import Cfg   # 更新后的总配置



def main():
    # load cfg
    cfg_parser = ConfigParse()
    args = cfg_parser.args

    # prepare before train
    default_setup(args)

    trainer = Trainer(args)
    trainer.train()




if __name__ == "__main__":
    main()



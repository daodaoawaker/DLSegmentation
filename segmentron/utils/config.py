import argparse

from segmentron.core.config import Cfg


class ConfigParse:
    def __init__(self):
        self.args = self.parse_args()
        self.cfg = Cfg
        self.update_cfg()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Segmentron config parse.")
        parser.add_argument('--config', type=str, default='config.yaml', required=True, 
                            help="path of config file")
        parser.add_argument('--seed', type=int, default=123456, 
                            help="random seed")
        parser.add_argument('--options', default=None, nargs=argparse.REMAINDER,
                            help="enable users to modify config options using command-lines")

        return parser.parse_args()

    def update_cfg(self):
        args = self.args

        self.cfg.defrost()
        self.cfg.merge_from_file(args.config)
        self.cfg.merge_from_list(args.options)
        self.cfg.freeze()

    # def get_cfg_defaults(self):
    #     return Cfg.clone()




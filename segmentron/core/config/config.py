import os
import argparse

from segmentron.core.config import defaultConfig


class ConfigParse:
    def __init__(self):
        self.args = self.parse_args()
        self.cfg = defaultConfig
        self.update_cfg()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Segmentron config parse.")
        
        parser.add_argument('--project', type=str, default='', required=True,
                            help="path of project")
        parser.add_argument('--config', type=str, default='config.yaml', required=True, 
                            help="path of config file")
        parser.add_argument('--seed', type=int, default=123456,
                            help="random seed")
        parser.add_argument('--local_rank', type=int, default=-1,
                            help="indicate current process index")
        parser.add_argument('options', default=None, nargs=argparse.REMAINDER,
                            help="enable users to modify config options using command-lines")

        return parser.parse_args()

    def update_cfg(self):
        args = self.args
        cfg = self.cfg

        cfg.defrost()
        cfg.merge_from_file(args.config)
        cfg.merge_from_list(args.options)

        cfg.log_dir = os.path.join(args.project, cfg.log_dir)
        cfg.copy_dir = os.path.join(args.project, cfg.copy_dir)
        cfg.output_dir = os.path.join(args.project, cfg.output_dir)

        cfg.freeze()

    # def get_cfg_defaults(self):
    #     return Cfg.clone()

Opt = ConfigParse()
UpdatedConfig = Opt.cfg




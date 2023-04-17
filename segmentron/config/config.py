import os
import torch
import argparse

from segmentron.config import defaultCfg
from tools import train, test


_Cfg = None

class ConfigParser:
    def __init__(self):
        self.cfg  = None
        self.args = None

        self.parser = None
        self.subparsers = None
        self.parser_test = None
        self.parser_train = None

        self.parsers()
        self.parse_args()
        self.update_cfg()

    def parsers(self):
        self._parser_master()
        self._parser_train()
        self._parser_test()

    def _parser_master(self):
        """main parser"""

        parser = argparse.ArgumentParser(description="Config parser")

        # parser.add_argument(
        #     '--project', 
        #     type=str, default='', required=True, help="path of project"
        # )
        # parser.add_argument(
        #     '--config', 
        #     type=str, default='config.yaml', required=True, help="path of config file"
        # )
        # parser.add_argument(
        #     '--seed', 
        #     type=int, default=123456, help="random seed"
        # )
        # parser.add_argument(
        #     '--local_rank', 
        #     type=int, default=-1, help="indicate current process index"
        # )
        # parser.add_argument(
        #     'options', 
        #     default=None, nargs=argparse.REMAINDER,
        #     help="enable users to modify config options using command-lines"
        # )

        self.parser = parser
        self.subparsers = self.parser.add_subparsers(help="sub-command help")

    def _parser_train(self):
        """subparser for train"""

        assert self.parser is not None
        assert self.subparsers is not None

        parser_train = self.subparsers.add_parser(
            "train", help="the entrance of Train pipeline"
        )
        parser_train.set_defaults(func=train)

        parser_train.add_argument(
            '--project', 
            type=str, default='', required=True, help="path of project"
        )
        parser_train.add_argument(
            '--config', 
            type=str, default='config.yaml', required=True, help="path of config file"
        )
        parser_train.add_argument(
            '--seed', 
            type=int, default=123456, help="random seed"
        )
        parser_train.add_argument(
            '--local_rank', 
            type=int, default=-1, help="indicate current process index"
        )
        parser_train.add_argument(
            'options', 
            default=None, nargs=argparse.REMAINDER,
            help="enable users to modify config options using command-lines"
        )
        parser_train.add_argument(
            '--num_workers',
            type=int, default=4, help=""    
        )

        self.parser_train = parser_train

    def _parser_test(self):
        """subparser for test"""

        assert self.parser is not None
        assert self.subparsers is not None

        parser_test = self.subparsers.add_parser(
            "test", help="the entrance of Test pipeline"
        )
        parser_test.set_defaults(func=test)

        parser_test = self.subparsers.add_parser(
            "train", help="the entrance of Train pipeline"
        )
        parser_test.set_defaults(func=train)
        parser_test.add_argument(
            '--project', 
            type=str, default='', required=True, help="path of project"
        )
        parser_test.add_argument(
            '--config', 
            type=str, default='config.yaml', required=True, help="path of config file"
        )
        parser_test.add_argument(
            '--seed', 
            type=int, default=123456, help="random seed"
        )
        parser_test.add_argument(
            '--local_rank', 
            type=int, default=-1, help="indicate current process index"
        )
        parser_test.add_argument(
            'options', 
            default=None, nargs=argparse.REMAINDER,
            help="enable users to modify config options using command-lines"
        )
        parser_test.add_argument(
            '--num_workers',
            type=int, default=4, help=""    
        )
        parser_test.add_argument(
            '--batch_size', 
            type=int, default=1, help="batch size of data"
        )

        self.parser_test = parser_test

    def parse_args(self):
        assert self.parser is not None
        assert self.subparsers is not None
        args = self.parser.parse_args()
        self.args = args

    def default_cfg(self):
        default_cfg = defaultCfg.clone()
        if self.cfg is None:
            self.cfg = default_cfg
        return default_cfg

    def update_cfg(self):
        global _Cfg

        self.default_cfg()
        assert self.cfg is not None
        assert self.args is not None

        cfg = self.cfg
        args = self.args

        cfg.defrost()
        cfg.merge_from_file(args.config)
        cfg.merge_from_list(args.options)

        cfg.log_dir = os.path.join(args.project, cfg.log_dir)
        cfg.copy_dir = os.path.join(args.project, cfg.copy_dir)
        cfg.output_dir = os.path.join(args.project, cfg.output_dir)

        cfg.freeze()
        _Cfg = cfg

    def run(self):
        args = self.args
        args.func(args)


Opt = ConfigParser()
Opt.args.task        =  _Cfg.TASK.TYPE
Opt.args.tester      =  _Cfg.TEST.TESTER
Opt.args.trainer     =  _Cfg.TRAIN.TRAINER
Opt.args.nprocs      =  torch.cuda.device_count()


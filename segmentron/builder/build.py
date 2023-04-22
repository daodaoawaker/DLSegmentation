
import pprint
from importlib import import_module
import torch.backends.cudnn as cudnn

from tools import snake2pascal
from segmentron.utils.utils import *
from segmentron.builder.loss import build_loss
from segmentron.utils.distributed import dist_init
from segmentron.utils.logger import Recorder
from segmentron.data import DataloaderBuilder
from segmentron.config import Cfg


class Builder:
    r'''Base class for train and test pipeline'''

    def __init__(self, args):
        local_rank = args.local_rank
        args.distributed = local_rank >= 0

        self._args = args
        self._model = None
        self._meta_arch = None
        self.num_gpus = args.nprocs
        self.local_rank = local_rank
        self.recorder = Recorder(args)
        self.logger = self.recorder.logger

        self.setup()
        self.config()
        self.create_meta_arch()
        self.criterion = build_loss(self._model)
        self.dataloader = DataloaderBuilder(args)

    @property
    def args(self):
        return self._args

    @property
    def meta_arch(self):
        return self._meta_arch

    def setup(self):
        # make directory if not exist
        if self.local_rank == 0:
            for dir in [Cfg.log_dir, Cfg.copy_dir, Cfg.output_dir]:
                make_if_not_exists(dir)
        # set seed for all random numbe r generator
        seed_for_all_rng(self.args.seed + self.local_rank)
        # initalize process group
        if self.args.distributed:
            cudnn.benchmark = Cfg.CUDNN.BENCHMARK
            cudnn.deterministic = Cfg.CUDNN.DETERMINISTIC
            cudnn.enabled = Cfg.CUDNN.ENABLED
            dist_init(self.args)

    def config(self):
        if self.local_rank == 0:
            self.logger.info(f"Using {self.num_gpus} GPUs.")
            self.logger.info(pprint.pformat(self.args))
            self.logger.info(Cfg)

    def create_meta_arch(self):
        task_type = self.args.task.lower()
        module_name = f'{task_type}_meta_arch'
        module = f'segmentron.apps.{task_type}.{module_name}'
        package = import_module(module)
        meta_arch_class = snake2pascal(module_name)
        meta_arch = getattr(package, meta_arch_class)()

        self._meta_arch = meta_arch
        self._model = meta_arch.model
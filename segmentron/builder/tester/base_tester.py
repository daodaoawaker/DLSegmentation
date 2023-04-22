
from segmentron.builder import Builder
from segmentron.builder.metric import get_metric
from segmentron.config import Cfg


class BaseTester(Builder):
    r"""Base class for test pipeline.

    There are the implementation of the general methods and attributes related to test.

    """
    def __init__(self, args):
        super(BaseTester, self).__init__(args)

        # Model
        self.test_loss = 0.0
        self.model = self.meta_arch.model

        # Data
        self.test_dataloader = self.dataloader.test_dataloader()

       # Distribution
        self.model_dist()

        # Metric
        self.mean_score = 0.0
        self.best_score = 0.0
        self.metric = get_metric(self.args)
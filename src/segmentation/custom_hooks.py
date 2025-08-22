import logging
import operator

import numpy as np
from detectron2.config import CfgNode
from detectron2.engine.hooks import HookBase

from src.segmentation.exceptions import EarlyStopperException


class EarlyStopper(HookBase):
    """
    Early Stopping Hook
    """

    def __init__(self, cfg: CfgNode):
        """
        Parameters
        ----------
        cfg : CfgNode
            Training config node containing EarlyStopping parameters.
        """
        self.__logger = logging.getLogger("segmentation_training")
        self.__patience = cfg.EARLY_STOPPING.PATIENCE
        self.__counter = 0
        self.__val_period = cfg.EARLY_STOPPING.VAL_PERIOD
        self.__val_metric = cfg.EARLY_STOPPING.METRIC

        assert cfg.EARLY_STOPPING.MODE in ["max", "min"], (
            f'Mode "{cfg.EARLY_STOPPING.MODE}" is not supported. Should be "max" or "min".'
        )
        if cfg.EARLY_STOPPING.MODE == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt

        self.best_metric = None
        self.best_iter = None

    def _update_best(self, val, iteration):
        """
        Update best metric and iteration.
        """
        self.__logger.debug("EarlyStopper: _update_best() called.")
        if np.isnan(val) or np.isinf(val):
            return False
        self.best_metric = val
        self.best_iter = iteration
        return True

    def _best_checking(self):
        """
        Check for improvement.
        """
        self.__logger.info(
            f"Early Stopper: Considering early stopping at patience {self.__counter}/{self.__patience}."
        )
        metric_tuple = self.trainer.storage.latest().get(self.__val_metric)
        if metric_tuple is None:
            self.__logger.warning(
                f"Early Stopper: Given validation metric {self.__val_metric} does not seem to be computed/stored. "
                "Will not be early stopping based on this metric."
            )
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if self.best_metric is None:
            self._update_best(latest_metric, metric_iter)
        elif self._compare(latest_metric, self.best_metric):
            self._update_best(latest_metric, metric_iter)
            self.__counter = 0
            self.__logger.info("Early Stopper: New best model found, resetting patience counter.")
        else:
            self.__counter += 1
            self.__logger.info(
                f"Early Stopper: Model did not improve since last {self.__counter} validation steps."
            )

    def after_step(self):
        """
        Implement early stopping using self._val_metric.
        """
        if self.trainer.iter > 0 and self.trainer.iter % self.__val_period == 0:
            self._best_checking()
            if self.__counter >= self.__patience:
                raise EarlyStopperException(
                    f"Early stopping triggered in iteration {self.trainer.iter}."
                )

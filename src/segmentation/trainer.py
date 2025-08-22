# -------------------------------------------------------
# Based on detectron2.engine.DefaultTrainer.
# Using ideas from detectron2/tools/plain_train_net.py
# -------------------------------------------------------

import logging
from collections import OrderedDict

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase, hooks
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.utils import comm
from detectron2.utils.events import EventStorage
from fvcore.nn.precise_bn import get_bn_modules
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from src.segmentation.custom_dataloader import npz_mapper
from src.segmentation.custom_hooks import EarlyStopper
from src.segmentation.exceptions import EarlyStopperException


class Trainer(DefaultTrainer):
    """
    Wrapper class for detectron2's DefaultTrainer.
    """

    def __init__(self, cfg: CfgNode):
        """
        Parameters
        ----------
        cfg : CfgNode
            Segmentation training cfg
        """
        super().__init__(
            cfg
        )  # This one has a ._trainer property which is used in a few places.

        self.start_iter = (
            self.checkpointer.resume_or_load(
                cfg.MODEL.WEIGHTS, resume=cfg.SOLVER.RESUME
            ).get("iteration", -1)
            + 1
        )
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.writers = self.build_writers()

    @classmethod
    def build_train_loader(cls, cfg: CfgNode) -> DataLoader:
        """
        Calls `detectron2.data.build_detection_train_loader` with `mapper = npz_mapper`.

        Parameters
        ----------
        cfg : CfgNode
            Segmentation training config node

        Returns
        -------
        DataLoader
        """
        return build_detection_train_loader(cfg, mapper=npz_mapper)  # type: ignore

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str) -> DataLoader:
        """
        Calls `detectron2.data.build_detection_test_loader` with `mapper = npz_mapper`.

        Parameters
        ----------
        cfg : CfgNode
            Segmentation training config node
        dataset_name : str
            Name of the dataset to be used (can be val or test)

        Returns
        -------
        DataLoader
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=npz_mapper)  # type: ignore

    @classmethod
    def build_evaluator(
        cls, cfg: CfgNode, dataset_name: str, fast: bool = False
    ) -> DatasetEvaluator:
        """
        Parameters
        ----------
        cfg : CfgNode
            Segmentation training config node
        dataset_name : str
            Name of the dataset used for evaluation (can be val or test)
        fast : bool
            From the COCOEvaluator docstring: 'use a fast but **unofficial** implementation to compute AP. Although the results should be very close to the official implementation in COCO API, it is still recommended to compute results with the official API for use in papers. The faster implementation also uses more RAM.'
            Used for validation only, where the time save is necessary.

        Returns
        -------
        COCOEvaluator
        """
        return (
            COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
            if not fast
            else COCOEvaluator(
                dataset_name, output_dir=cfg.OUTPUT_DIR, use_fast_impl=True
            )
        )

    def build_hooks(self) -> list[HookBase]:
        """
        Overwrite `detectron2.engine.DefaultTrainer.build_hooks` for the purpose of detailed documentation.

        "Build a list of default hooks, including timing, evaluation, checkpointing, lr scheduling, precise BN, writing events.

        Returns
        -------
        list[HookBase]
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = (
            0  # @ORIGINAL_COMMENT: save some memory and time for PreciseBN
        )

        ret = [
            hooks.IterationTimer(),  # Track the time spent for each iteration and print summary at the end of training.
            hooks.LRScheduler(),  # Use torch builtin Learning Rate scheduler.
            hooks.PreciseBN(  # Improve BatchNorm
                # @ORIGINAL_COMMENT: Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,  # pyright: ignore[reportAttributeAccessIssue]
                # @ORIGINAL_COMMENT: Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)  # pyright: ignore[reportAttributeAccessIssue]
            else None,
        ]

        # @ORIGINAL_COMMENT:
        # Do PreciseBN before checkpointer, because it updates the model and needs to be
        # saved by the checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():  # This only affects runs using multiple GPUs
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=1
                )
            )

        self._last_eval_results = None

        def val_and_save_results():
            self._last_eval_results = self.validate(self.cfg, self.model)  # pyright: ignore[reportAttributeAccessIssue]
            return self._last_eval_results

        # @ORIGINAL_COMMENT:
        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # Add validation Hook, then BestCheckpointer and EarlyStopper (depending on cfg).
        ret.append(
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD, val_and_save_results, eval_after_train=False
            )
        )
        if comm.is_main_process():
            ret.append(
                hooks.BestCheckpointer(
                    cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP", "max", "best"
                )
            )
            if cfg.EARLY_STOPPING.ENABLED:
                ret.append(EarlyStopper(cfg))

        # Finally, add test hook
        self.test_results = None

        def test_and_save_results():
            self.test_results = self.test(self.cfg, self.model)  # pyright: ignore[reportAttributeAccessIssue]
            return self.test_results

        ret.append(hooks.EvalHook(0, test_and_save_results, eval_after_train=True))

        if comm.is_main_process():
            # @ORIGINAL_COMMENT: Here the default print/log frequency of each writer is used.
            # run wrtiers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=100))

        return ret

    def train(self):
        """
        Copy of train() from TrainerBase for catching EarlyStopperException.
        """
        start_iter = self.start_iter
        max_iter = self.max_iter

        logger = logging.getLogger(__name__)
        logger.info(f"Starting training from iteration {start_iter}")

        self.iter = self.start_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except EarlyStopperException as msg:
                logger.info(f"{msg} Reverting model to best checkpoint.")
                # Revert Model zu current best.pth
                DetectionCheckpointer(self.model).load(  # pyright: ignore[reportAttributeAccessIssue]
                    self.cfg.OUTPUT_DIR + "/best.pth"
                )
                self.max_iter = self.iter - 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(self, "_last_eval_results"), (
                "No evaluation results obtained during training!"
            )
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def validate(cls, cfg, model, evaluators=None):
        """
        Same as test(), but for validation set
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.VAL) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.VAL), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.VAL):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    # Using the fast implementation saves about 20% time in evaluation.
                    evaluator = cls.build_evaluator(cfg, dataset_name, fast=True)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(results_i, dict), (
                    "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        return results

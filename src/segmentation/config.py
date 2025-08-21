import os
import sys
from pathlib import Path

import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from yacs.config import CfgNode

CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())


def config(global_cfg: CfgNode, n_train: int, resume: bool = False) -> CfgNode:
    """
    Yields the training config for detectron2

    Parameters
    ----------
    global_cfg : CfgNode
        The global config read from file, partially merged into the training config
    n_train : int
        The number of instances in the training set. Used to calculate epochs.
    resume : bool
        Whether to resume training from a checkpoint.

    Returns
    -------
    CfgNode
        The training config for detectron2.

    NOTES
    -----
    - The model configuration is already using the smallest out-of-the-box model available.
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TRAIN = global_cfg.SEGMENTATION.DATASETS.TRAIN
    cfg.DATASETS.TEST = global_cfg.SEGMENTATION.DATASETS.TEST
    cfg.DATASETS.VAL = global_cfg.SEGMENTATION.DATASETS.VAL

    cfg.DATALOADER.NUM_WORKERS = global_cfg.SEGMENTATION.TRAINING.n_workers

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        global_cfg.SEGMENTATION.MODEL.ROI_per_image
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = global_cfg.SEGMENTATION.MODEL.ROI_num_classes

    cfg.SOLVER.IMS_PER_BATCH = global_cfg.SEGMENTATION.TRAINING.batch_size
    cfg.SOLVER.BASE_LR = global_cfg.SEGMENTATION.TRAINING.lr
    cfg.SOLVER.MAX_ITER = (
        int(
            np.ceil(
                (n_train / cfg.SOLVER.IMS_PER_BATCH)
                * global_cfg.SEGMENTATION.TRAINING.max_epochs
            )
        )
        if global_cfg.SEGMENTATION.TRAINING.max_epochs != 0
        else 0
    )
    cfg.SOLVER.CHECKPOINT_PERIOD = (
        int(
            np.ceil(
                cfg.SOLVER.MAX_ITER
                / (
                    global_cfg.SEGMENTATION.TRAINING.max_epochs
                    / global_cfg.SEGMENTATION.TRAINING.checkpoint_every
                )
            )
        )
        if global_cfg.SEGMENTATION.TRAINING.max_epochs != 0
        else 0
    )
    cfg.SOLVER.RESUME = resume
    cfg.SOLVER.AMP.ENABLED = False

    cfg.TEST.EVAL_PERIOD = (
        int(
            np.ceil(
                cfg.SOLVER.MAX_ITER
                / (
                    global_cfg.SEGMENTATION.TRAINING.max_epochs
                    / global_cfg.SEGMENTATION.TRAINING.val_every
                )
            )
        )
        if global_cfg.SEGMENTATION.TRAINING.max_epochs != 0
        else 0
    )

    cfg.EARLY_STOPPING = CfgNode()
    cfg.EARLY_STOPPING.ENABLED = global_cfg.SEGMENTATION.EARLY_STOPPING.enable
    cfg.EARLY_STOPPING.PATIENCE = global_cfg.SEGMENTATION.EARLY_STOPPING.patience
    cfg.EARLY_STOPPING.VAL_PERIOD = (
        int(
            np.ceil(
                cfg.SOLVER.MAX_ITER
                / (
                    global_cfg.SEGMENTATION.TRAINING.max_epochs
                    / global_cfg.SEGMENTATION.EARLY_STOPPING.interval
                )
            )
        )
        if global_cfg.SEGMENTATION.TRAINING.max_epochs != 0
        else 0
    )
    cfg.EARLY_STOPPING.METRIC = global_cfg.SEGMENTATION.EARLY_STOPPING.metric
    cfg.EARLY_STOPPING.MODE = global_cfg.SEGMENTATION.EARLY_STOPPING.mode

    cfg.OUTPUT_DIR = global_cfg.SEGMENTATION.TRAINING.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.SEED = global_cfg.SEGMENTATION.TRAINING.seed

    cfg.freeze()

    return cfg


def inf_config(global_cfg: CfgNode):
    """
    Yield inference config for detectron2
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TEST = global_cfg.SEGMENTATION.DATASETS.TEST

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        global_cfg.SEGMENTATION.MODEL.ROI_per_image
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = global_cfg.SEGMENTATION.MODEL.ROI_num_classes

    cfg.EARLY_STOPPING = CfgNode()
    cfg.EARLY_STOPPING.ENABLED = False

    cfg.SOLVER.RESUME = False

    cfg.OUTPUT_DIR = global_cfg.SEGMENTATION.TRAINING.output_dir
    cfg.MODEL.WEIGHTS = os.path.join(CWD, cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.DEVICE = "cuda:1"

    cfg.freeze()

    return cfg

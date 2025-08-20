import os
import sys
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

# find top directory
CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())

from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.engine.launch import launch  # noqa: E402
from yacs.config import CfgNode  # noqa: E402

from src import log  # noqa: E402
from src.data.dataset_registration import register_dataset_json  # noqa: E402
from src.segmentation.config import config  # noqa: E402
from src.segmentation.trainer import Trainer  # noqa: E402
from src.segmentation.utils import seed_everything  # noqa: E402


def _train_segmentation_model(global_cfg: CfgNode, resume: bool = False) -> None:
    """
    Train Mask-RCNN segmentation model
    """
    lggr = getLogger("segmentation_training")
    lggr = log.log_only_console(lggr)

    lggr.info("Registering datasets...")
    register_dataset_json(global_cfg.DATA.path)

    n_train = len(DatasetCatalog.get("raven-f_train"))

    cfg = config(global_cfg, n_train, resume)

    seed_everything(cfg.SEED)

    lggr.info(f"Found {n_train} training instances.")

    trainer = Trainer(cfg)

    trainer.train()
    trainer.test(cfg, trainer.model) # pyright: ignore[reportAttributeAccessIssue]

    lggr.info("Training complete.")


def train_segmentation_model():
    """
    Call _train_segmentation_model with the appropriate config and launch.
    When calling this script directly, you can use the '-r' or '--resume' flags to enable continuation from the last checkpoint.
    """
    prsr = ArgumentParser(description="Train Mask-RCNN segmentation model")

    prsr.add_argument(
        "-r",
        "--resume",
        action="store_true",
        default=False,
        required=False,
        help="Pass this flag to resume training from the latest checkpoint.",
    )

    args = prsr.parse_args()

    with open(os.path.join(CWD, "global_cfg.yml"), "r") as f:
        global_cfg = CfgNode.load_cfg(f)

    launch(
        _train_segmentation_model,
        num_gpus_per_machine=global_cfg.MACHINE.num_gpus,
        dist_url="auto",
        args=(global_cfg, args.resume),
    )


if __name__ == "__main__":
    log.init()
    train_segmentation_model()

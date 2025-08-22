import logging
import os
import sys
from pathlib import Path
from shutil import copyfile

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from yacs.config import CfgNode

CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())
sys.path.append(os.path.join(CWD, "src", "representation", "PyTorch_VAE"))

from src import log  # noqa: E402
from src.data.RavenShapes import RavenShapesDataModule  # noqa: E402
from src.representation.bTCVAExperiment import bTCVAExperiment  # noqa: E402
from src.representation.PyTorch_VAE.models import *  # noqa: E402, F403
from src.representation.PyTorch_VAE.utils import seed_everything  # noqa: E402


def train_vae():
    """ """
    logger = logging.getLogger("vae_training")

    with open(os.path.join(CWD, "global_cfg.yml"), "r") as f:
        global_cfg = CfgNode.load_cfg(f)

    # Tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=global_cfg.REPRESENTATION.TRAINING.output,
        name=global_cfg.REPRESENTATION.MODEL.name,
    )

    Path(os.path.join(tb_logger.log_dir, "Samples")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(tb_logger.log_dir, "Reconstructions")).mkdir(
        parents=True, exist_ok=True
    )

    seed_everything(global_cfg.REPRESENTATION.TRAINING.seed, deterministic=False)

    # Model
    model = vae_models[global_cfg.REPRESENTATION.MODEL.name](  # noqa: F405
        hidden_dims=[32] * int(np.log2(global_cfg.REPRESENTATION.TRAINING.patch_size / 4)),
        **global_cfg.REPRESENTATION.MODEL,
    )
    experiment = bTCVAExperiment(
        model,
        params={
            "LR": global_cfg.REPRESENTATION.TRAINING.lr,
            "weight_decay": global_cfg.REPRESENTATION.TRAINING.weight_decay,
            "scheduler_gamma": global_cfg.REPRESENTATION.TRAINING.scheduler_gamma,
            "kld_weight": global_cfg.REPRESENTATION.TRAINING.kld_weight,
            "manual_seed": global_cfg.REPRESENTATION.TRAINING.seed,
            "deterministic": False,
        },
    )

    # Data
    data = RavenShapesDataModule(global_cfg, global_cfg.MACHINE.num_gpus > 0)
    data.setup()

    # Trainer
    trainer = L.Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            (
                checkpointer := ModelCheckpoint(
                    save_top_k=1,
                    dirpath=os.path.join(
                        tb_logger.log_dir, "checkpoints"
                    ),
                    monitor="val_loss",
                    save_last=True,
                )
            ),
        ],
        strategy=DDPStrategy(),
        log_every_n_steps=25,
        **{
            "accelerator": "gpu" if torch.cuda.is_available() else None,
            "devices": list(range(global_cfg.MACHINE.num_gpus))
            if torch.cuda.is_available()
            else None,
            "max_epochs": global_cfg.REPRESENTATION.TRAINING.max_epochs,
        },
    )

    logger.info(f"Training VAE model: {global_cfg.REPRESENTATION.MODEL.name}")
    trainer.fit(experiment, data)
    logger.info("Training Finished.")

    # rename best checkpoint
    try:
        os.rename(
            checkpointer.best_model_path,
            os.path.join(
                global_cfg.REPRESENTATION.TRAINING.output,
                "checkpoints",
                f"best_{global_cfg.REPRESENTATION.MODEL.name}.ckpt",
            ),
        )
    except FileNotFoundError:
        pass

    # cleanup
    for subdir in Path(
        os.path.join(
            global_cfg.REPRESENTATION.TRAINING.output,
            global_cfg.REPRESENTATION.MODEL.name,
        )
    ).iterdir():
        if subdir.is_dir():
            if subdir.joinpath("checkpoints") in subdir.iterdir():
                continue
            else:
                os.system(f"rm -rf {subdir}")
                print(
                    f"Removed directory: {subdir} artifact resulting form multiprocessing."
                )


if __name__ == "__main__":
    log.init()
    train_vae()

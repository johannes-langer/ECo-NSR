import os
import shutil
import sys
from pathlib import Path

import splitfolders
from RAVEN_FAIR.src.main import main as raven_fair_main
from yacs.config import CfgNode

# find top directory
CWD = [p for p in Path(__file__).parents if p.stem == 'eco-nsr'][0]
sys.path.append(CWD.as_posix())

def generate_raven():
    """
    Generate the RAVEN-FAIR dataset to the directory specified in global_cfg.
    """

    with open(os.path.join(CWD, 'global_cfg.yml'), 'r') as f:
        cfg = CfgNode.load_cfg(f)

    # compile cfg into command line arguments for the RAVEN-F generator
    argv = [
        '--num-samples', str(cfg.DATA.RAVEN.num_samples),
        '--save-dir', str(cfg.DATA.path),
        '--seed', str(cfg.DATA.RAVEN.seed),
        '--fair', str(cfg.DATA.RAVEN.fair),
        '--val', str(cfg.DATA.RAVEN.val),
        '--test', str(cfg.DATA.RAVEN.test),
        '--save', str(cfg.DATA.RAVEN.save),
        '--cocoapi', str(cfg.DATA.RAVEN.cocoapi)
    ]

    # call generator
    raven_fair_main(argv)

    # directory split
    splitfolders.ratio(
        input = os.path.join(cfg.DATA.path, 'RAVEN-F'),
        output = os.path.join(cfg.DATA.path, '_RAVEN-F'),
        seed = cfg.DATA.RAVEN.seed,
        ratio = (cfg.DATA.RAVEN.train_ratio, cfg.DATA.RAVEN.val_ratio, cfg.DATA.RAVEN.test_ratio),
        group_prefix = 2,
        move = True
    )

    # clean up
    shutil.rmtree(os.path.join(cfg.DATA.path, 'RAVEN-F'))
    os.rename(os.path.join(cfg.DATA.path, '_RAVEN-F'), os.path.join(cfg.DATA.path, 'RAVEN-F'))

if __name__ == '__main__':
    generate_raven()

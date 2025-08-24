from copy import deepcopy
from io import StringIO
import os
import sys
from logging import getLogger
from pathlib import Path
from typing import Union

import numpy as np
from yacs.config import CfgNode

CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())
sys.path.append(os.path.join(CWD, "src", "representation", "PyTorch_VAE"))

from src.representation.cluster import cluster  # noqa: E402
from src.representation.PyTorch_VAE.models import BaseVAE  # noqa: E402


def to_popper(
    model: BaseVAE,
    matrix: list[dict],
    matrix_idx: int,
    global_cfg: CfgNode,
    strategy: str = "trial",
    path: Union[str, Path, None] = None,
) -> None:
    """
    Convert ``matrix`` to a Popper-compatible format using the trained representation ``model``.

    Parameters
    ----------
    model : BaseVAE
        The trained VAE model used for conversion.
    matrix : list[dict]
        The input matrix to be converted, in the format produced by the RavenMatrices ``Dataset``.
    matrix_idx : int
        The matrix index based on the RavenMatrices dataset file.
    global_cfg : CfgNode
        From global_cfg.yml
    strategy : str
        The strategy used to assemble the example files:
        - "minimal": Only use the first lines in the matrix as positive examples and no negative examples
        - "noisy": Use all answer options as negative examples, expect that the best program found will still cover the true answer.
        - "trial": Produce multiple example files, make only one option positive in each of them. Expect that only one file can learn a rule with 100% coverage.
    path : Union[str, Path, None], optional
        The path to save the converted files to. Should be a directory, where the files bk.pl, bias.pl, and exs.pl will be saved. If not provided, use global_cfg to determine the path.
    """
    if strategy not in ["minimal", "noisy", "trial"]:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if path is None:
        path = os.path.join(CWD, global_cfg.REASONING.POPPER.file_path, f"matrix_{matrix_idx}")

    os.makedirs(path, exist_ok=True)

    logger = getLogger("to_popper")

    discrete_representation = cluster(model, matrix, global_cfg)

    true_target = [
        idx - 8 for idx, panel in enumerate(matrix) if panel["type"] == "target"
    ][0]
    logger.info(f"True target is option {true_target}")

    # --- background knowledge ---

    logger.info("Creating background knowledge.")

    with open(os.path.join(path, "bk.pl"), "w") as bk:

        with open(os.path.join(CWD, "src", "reasoning", "base_bk.pl"), "r") as f:
            bk.write(f.read())

        answer_counter = 0
        shape_counter = 0
        for panel in matrix:
            # get a panel id
            panel_id = (
                panel["type"] if "task" in panel["type"] else f"answer{answer_counter}"
            )
            answer_counter += 1 if "answer" in panel_id else 0

            # write meta information
            bk.write(f"n_shapes({panel_id}, {len(panel['locs'])}).\n")

            for idx in range(len(panel["locs"])):
                bk.write(f"has_shape({panel_id}, {panel_id}_{idx}).\n")

                # write latent features
                for dim in range(discrete_representation.shape[1]):
                    bk.write(
                        f"shape_prop({panel_id}_{idx}, latent{dim}, {discrete_representation[shape_counter, dim].item()}).\n"
                    )

                # write location information
                loc = panel["locs"][idx]
                loc_x = int(
                    np.floor(
                        (loc[0] + (loc[2] - loc[0]) / 2)
                        / global_cfg.REASONING.loc_grid_size
                    )
                )
                loc_y = int(
                    np.floor(
                        (loc[1] + (loc[3] - loc[1]) / 2)
                        / global_cfg.REASONING.loc_grid_size
                    )
                )

                bk.write(f"shape_prop({panel_id}_{idx}, loc_x, {loc_x}).\n")
                bk.write(f"shape_prop({panel_id}_{idx}, loc_y, {loc_y}).\n")

                # write location information
                loc = panel["locs"][idx]
                loc_x = int(
                    np.floor(
                        (loc[0] + (loc[2] - loc[0]) / 2)
                        / global_cfg.REASONING.loc_grid_size
                    )
                )
                loc_y = int(
                    np.floor(
                        (loc[1] + (loc[3] - loc[1]) / 2)
                        / global_cfg.REASONING.loc_grid_size
                    )
                )

                bk.write(f"shape_prop({panel_id}_{idx}, loc_x, {loc_x}).\n")
                bk.write(f"shape_prop({panel_id}_{idx}, loc_y, {loc_y}).\n")

                shape_counter += 1

    # --- examples ---

    logger.info(f"Creating examples using the '{strategy}' strategy.")

    if strategy == "trial":
        for i in range(answer_counter):
            with open(os.path.join(path, f"exs_option_{i}.pl"), "w") as exs:
                exs.write("pos(row(task0, task1, task2)).\n")
                exs.write("pos(row(task3, task4, task5)).\n")
                for j in range(answer_counter):
                    if i != j:
                        exs.write(f"neg(row(task6, task7, answer{j})).\n")
                    else:
                        exs.write(f"pos(row(task6, task7, answer{j})).\n")
    
    else:
        with open(os.path.join(path, "exs.pl"), "w") as exs:

            # the first two are needed regardless of strategy
            exs.write("pos(row(task0, task1, task2)).\n")
            exs.write("pos(row(task3, task4, task5)).\n")

            if strategy == "noisy":
                for i in range(answer_counter):
                    exs.write(f"neg(row(task6, task7, answer{i})).\n")

    # --- bias ---

    with open(os.path.join(path, "bias.pl"), "w") as bias:
        with open(os.path.join(CWD, "src", "reasoning", "base_bias.pl"), "r") as f:
            bias.write(f.read())

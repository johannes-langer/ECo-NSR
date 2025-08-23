from copy import deepcopy
import os
import sys
from logging import getLogger
from pathlib import Path
from io import StringIO
from turtle import st
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
    global_cfg: CfgNode,
    strategy: str = "trial",
    path: Union[str, Path, None] = None,
    save: bool = False,
) -> tuple[
    StringIO,
    StringIO,
    Union[StringIO, list[StringIO]],
]:
    """
    Convert ``matrix`` to a Popper-compatible format using the trained representation ``model``.

    Parameters
    ----------
    model : BaseVAE
        The trained VAE model used for conversion.
    matrix : list[dict]
        The input matrix to be converted, in the format produced by the RavenMatrices ``Dataset``.
    global_cfg : CfgNode
        From global_cfg.yml
    strategy : str
        The strategy used to assemble the example files:
        - "minimal": Only use the first lines in the matrix as positive examples and no negative examples
        - "noisy": Use all answer options as negative examples, expect that the best program found will still cover the true answer.
        - "trial": Produce multiple example files, make only one option positive in each of them. Expect that only one file can learn a rule with 100% coverage.
    path : Union[str, Path, None], optional
        The path to save the converted files to. Should be a directory, where the files bk.pl, bias.pl, and exs.pl will be saved.
    save : bool, optional
        Whether to save the converted files. If True, ``path`` must be provided.

    Returns
    -------
    tuple[SpooledTemporaryFile, SpooledTemporaryFile, Union[SpooledTemporaryFile, list[SpooledTemporaryFile]]]
        A tuple containing the converted files: bk, bias, and exs. This function uses temporary files in memory. Use ``save = True`` to save them to disk. exs can be a single file or a list of files, depending on ``strategy``.
    """
    if save and path is None:
        raise ValueError("If save is True, path must be provided.")

    if strategy not in ["minimal", "noisy", "trial"]:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger = getLogger("to_popper")

    discrete_representation = cluster(model, matrix, global_cfg)

    true_target = [
        idx - 8 for idx, panel in enumerate(matrix) if panel["type"] == "target"
    ][0]
    logger.info(f"True target is option {true_target}")

    # --- background knowledge ---

    logger.info("Creating background knowledge.")

    bk = StringIO()

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
            loc_x = int(np.floor((loc[0] + (loc[2] - loc[0]) / 2) / global_cfg.REASONING.loc_grid_size))
            loc_y = int(np.floor((loc[1] + (loc[3] - loc[1]) / 2) / global_cfg.REASONING.loc_grid_size))

            bk.write(f"shape_prop({panel_id}_{idx}, loc_x, {loc_x}).\n")
            bk.write(f"shape_prop({panel_id}_{idx}, loc_y, {loc_y}).\n")

            # write location information
            loc = panel["locs"][idx]
            loc_x = int(np.floor((loc[0] + (loc[2] - loc[0]) / 2) / global_cfg.REASONING.loc_grid_size))
            loc_y = int(np.floor((loc[1] + (loc[3] - loc[1]) / 2) / global_cfg.REASONING.loc_grid_size))

            bk.write(f"shape_prop({panel_id}_{idx}, loc_x, {loc_x}).\n")
            bk.write(f"shape_prop({panel_id}_{idx}, loc_y, {loc_y}).\n")

            shape_counter += 1

    bk.seek(0)

    # --- examples ---

    logger.info(f"Creating examples using the '{strategy}' strategy.")

    exs = StringIO()

    # the first two are needed regardless of strategy
    exs.write("pos(row(task0, task1, task2)).\n")
    exs.write("pos(row(task3, task4, task5)).\n")

    if strategy == "minimal":
        exs.seek(0)

    if strategy == "noisy":
        for i in range(answer_counter):
            exs.write(f"neg(row(task6, task7, answer{i})).\n")
        exs.seek(0)

    if strategy == "trial":
        ret = []
        for i in range(answer_counter):
            _exs = deepcopy(exs)
            for j in range(answer_counter):
                if i != j:
                    _exs.write(f"neg(row(task6, task7, answer{j})).\n")
                else:
                    _exs.write(f"pos(row(task6, task7, answer{j})).\n")
            ret.append(_exs.seek(0))
        exs = ret

    # --- bias ---

    bias = StringIO()
    with open(os.path.join(CWD, "src", "reasoning", "base_bias.pl"), "r") as f:
        bias.write(f.read())
    bias.seek(0)

    return bk, bias, exs

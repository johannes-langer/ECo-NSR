import os
import sys
from pathlib import Path

import torch
from sklearn.cluster import MeanShift
from torchvision.transforms import Resize
from yacs.config import CfgNode

CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())
sys.path.append(os.path.join(CWD, "src", "representation", "PyTorch_VAE"))

from src.reasoning.utils import crop_img  # noqa: E402
from src.representation.PyTorch_VAE.models import BaseVAE  # noqa: E402


def cluster(
    model: BaseVAE,
    matrix: list[dict],
    global_cfg: CfgNode,
) -> torch.Tensor:
    """
    Produce discrete latent representation by clustering over all continuous representations of shapes in the matrix.

    Parameters
    ----------
    model : BaseVAE
        The trained VAE model used for conversion.
    matrix : list[dict]
        The input matrix to be converted, in the format produced by the RavenMatrices ``Dataset``.
    global_cfg : CfgNode
        From global_cfg.yml

    Returns
    -------
    torch.Tensor
        A tensor of shape ``[n_shapes, latent_dim]`` where ``n_shapes`` is the total number of shapes in the matrix and ``latent_dim`` is the dimensionality of the latent space of the VAE. Values are discrete cluster membership.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # collect all shapes as one batched tensor
    all_shapes = torch.tensor(0)
    for panel in matrix:
        img = panel["img"]
        for box in panel["locs"]:
            shape = Resize((
                global_cfg.REPRESENTATION.TRAINING.patch_size,
                global_cfg.REPRESENTATION.TRAINING.patch_size,
            ))(crop_img(img, box)).unsqueeze(0)
            all_shapes = (
                shape
                if all_shapes.dim() == 0
                else torch.cat((all_shapes, shape), dim=0)
            )

    # embed
    model.eval()
    model.to(device)
    with torch.no_grad():
        mu, logvar = model.encode(all_shapes.to(device))
        z = model.reparameterize(mu, logvar).cpu()  # pyright: ignore[reportCallIssue]; All BaseVAE subclasses implement reparameterize. Can be understood as sampling from a normal distribution parameterized by mu and logvar.

    for dim in range(global_cfg.REPRESENTATION.MODEL.latent_dim):
        clustering = MeanShift().fit(z[:, dim].reshape(-1, 1))
        clusters = (
            torch.tensor(clustering.labels_).unsqueeze(1)
            if dim == 0
            else torch.cat(
                (clusters, torch.tensor(clustering.labels_).unsqueeze(1)),  # pyright: ignore[reportPossiblyUnboundVariable]  # noqa: F821
                dim=1,
            )
        )

    return clusters  # pyright: ignore[reportPossiblyUnboundVariable]

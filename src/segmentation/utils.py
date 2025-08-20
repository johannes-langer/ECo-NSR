import os
import random
import numpy as np
import torch


def get_img_size(path: str, image_id: int) -> tuple:
    """
    Returns size of an image contained in an .npz file from a RAVEN dataset
    """
    with np.load(path) as file:
        out = file["image"][image_id].shape

    return out


def get_panel_from_id(id: str) -> int:
    """
    Takes string id of the format fileid_panelid and returns panelid as an integer
    """
    return int(id.split("_")[1])


def get_panel_image(instance: dict) -> np.ndarray:
    """
    returns the image from the dict representing the instance. Expects the found image to be grayscale and returns rgb

    Parameters
    ----------
    instance : dict
    """
    panel_id = get_panel_from_id(instance["image_id"])
    with np.load(instance["file_name"]) as file:
        img_array = file["image"][panel_id]
    return np.tile(img_array, (3, 1, 1)).transpose([1, 2, 0])

def seed_everything(seed : int) -> None:
    """
    Set all random seeds to ensure reproducibility.
    """    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
import numpy as np


def crop_img(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Use bbox to crop an image to a single panel.

    Parameters
    ----------
    img : np.ndarray
        Panel image of shape [channels, height, width]
    bbox : list[float]
        Shape bounding box

    Returns
    -------
    np.ndarray
        img cropped to the specified bounding box
    """
    if len(img.shape) == 3:
        return img[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
    if len(img.shape) == 4:
        return img[:, :, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
    else:
        raise ValueError("Invalid image shape")
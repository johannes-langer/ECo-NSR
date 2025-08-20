import copy

import torch
from detectron2.data import detection_utils

from src.segmentation import utils


def npz_mapper(dataset_dict : dict) -> dict:
    """
    Mapper for RAVEN datasets.

    Parameters
    ----------
    dataset_dict : dict

    Returns
    -------
    dict
    """
    dataset_dict = copy.deepcopy(dataset_dict)

    # Read image
    image = utils.get_panel_image(dataset_dict)
    detection_utils.check_image_size(dataset_dict, image)

    image = torch.from_numpy(image.transpose(2, 0, 1))

    annos = [
        detection_utils.transform_instance_annotations(annotation, [], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]

    return {
        "image": image,
        "instances": detection_utils.annotations_to_instances(
            annos, tuple(image.shape[1:])
        ),
        "height": image.shape[1],
        "width": image.shape[2],
        "image_id": dataset_dict["image_id"],
    }


def inference_mapper(dataset_dict : dict) -> dict:
    """
    Inference Mapper

    Parameters
    ----------
    dataset_dict : dict

    Returns
    -------
    dict
    """
    image = utils.get_panel_image(dataset_dict)
    return {
        "image": image,
        "image_id": dataset_dict["image_id"],
    }

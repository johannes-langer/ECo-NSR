import ast
import os
from pathlib import Path
import sys

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from tqdm import tqdm

from src.data.utils import convert_stringlist, get_xml


# find top directory
CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())

def register_dataset_raw(root_path: str) -> None:
    """
    Registers the RAVEN-F dataset to detectron2's DatasetCatalog and MetadataCatalog.

    Datasets can be accessed as: `<raven/raven-f>_<train/val/test>`

    Parameters
    ----------
    root_path : str
        Absolute path to dataset directory (not including "RAVEN-F"!!!)
    """
    for d in ["train", "val", "test"]:
        DatasetCatalog.register(
            "raven-f_" + d,
            lambda d=d: format_raven(
                path=root_path, split=d
            ),  # lambda expression necessary for this to be treated as simply a function name
        )
        MetadataCatalog.get("raven-f_" + d).set(thing_classes=["object"])


def register_dataset_json(root_path: str) -> None:
    """
    Registers the RAVEN-F dataset to detectron2's DatasetCatalog and MetadataCatalog.

    Parameters
    ----------
    root_path : str
        Absolute path to dataset directory (not including "RAVEN-F"!!!)
    """
    for split in ["train", "val", "test"]:
        register_coco_instances(
            "raven-f_" + split,
            {},
            os.path.join(os.path.join(CWD, 'outputs', 'data'), f"raven-f_{split}.json"),
            root_path
        )
        MetadataCatalog.get("raven-f_" + split).set(thing_classes=["object"])


def format_raven(path: str, split: str = "train") -> list[dict]:
    """
    dataset function providing `list[dict]` with a specification similar COCO's annotations.

    Parameters
    ----------
    path : str
        Absolute path to dataset directory (not including "RAVEN-F"!!!)
    split : str
        The dataset split (train, test, val)

    Returns
    -------
    list[dict]
        in COCO format
    """
    # Pick Root Directory for dataset
    path = os.path.join(path, "RAVEN-F", split)

    # Use pathlib to create a generator yielding all .xml files in path (recursive)
    files = Path(path).rglob("*.xml")
    total_iterations = len(
        list(files)
    )  # This overwrites files, which is why I re-define it in the next line
    files = Path(path).rglob("*.xml")  # essentially, these two lines make tqdm work

    # Begin loop level 1 over files
    dataset_dicts = []
    for file_id, file in tqdm(enumerate(files), total=total_iterations):
        # Load Metadata of file:
        file_string = file.as_posix()
        file_dict = get_xml(file_string)
        corresp_data = file_string[:-4] + ".npz"

        # how many images are there (should usually be 16):
        with np.load(corresp_data) as npz:
            n_imgs = len(npz["image"])
        try:
            assert n_imgs == 16, f"file {file_string} does not contain 16 images"
        except AssertionError as e:
            print(e)

        # Begin loop level 2 over panels
        for img_id in range(n_imgs):
            # Create dict for this entry
            record = {}

            # Get image
            with np.load(corresp_data) as npz:
                img = npz["image"][img_id]

            # Get image size
            width, height = img.shape

            # Common
            record["file_name"] = corresp_data
            record["image_id"] = f"{file_id}_{img_id}"
            record["height"] = height
            record["width"] = width

            # Segmentation
            panel = file_dict["Data"]["Panels"]["Panel"][img_id]
            components = panel["Struct"]["Component"]

            components = (
                components if type(components) is list else [components]
            )  # This makes sure we can iterate over components, even if it only has a single entry.

            # Begin loop level 3 over objects
            objs = []
            for cmp in components:
                entities = cmp["Layout"]["Entity"]
                entities = entities if type(entities) is list else [entities]

                # Begin loop level 4, at the same semantic level as level 3
                for ent in entities:
                    # get bounding box as lists
                    real_bbox = convert_stringlist(ent["@real_bbox"])

                    # calculate bounding box

                    center_x = np.ceil(real_bbox[1] * width)
                    center_y = np.ceil(real_bbox[0] * height)
                    ent_width = np.ceil(real_bbox[3] * width)
                    ent_height = np.ceil(real_bbox[2] * height)
                    bbox = [
                        np.floor(center_x - 0.5 * ent_width),
                        np.floor(center_y - 0.5 * ent_height),
                        np.ceil(center_x + 0.5 * ent_width),
                        np.ceil(center_y + 0.5 * ent_height),
                    ]

                    seg = [
                        (x + 0.5, y + 0.5) for x, y in ast.literal_eval(ent["@mask"])
                    ]
                    seg = [p for x in seg for p in x]

                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [seg],
                        "category_id": 0,
                    }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

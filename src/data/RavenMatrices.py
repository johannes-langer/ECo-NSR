import ast
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from yacs.config import CfgNode

CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())

from src import log  # noqa: E402
from src.data.utils import get_xml  # noqa: E402

# --- Classes ---


class RavenMatrices(Dataset):
    """
    Torch Dataset Class for RAVEN(-F) full tasks, meaning it produces a set of panels which dictate the task.

    This dataset returns lists of dicts with the following keys:
    - `img` (Tensor) : The image of the panel.
    - `masks` (list[Tensor]) : A list of masks, one for each shape in the panel.
    - `locs` (list[ndarray]) : A list of location markers, one for each shape in the panel. Each is [center_x, center_y, height, width]
    - `type` (str) : One of `task[0-7]`, `target`, `distractor`.

    """

    def __init__(
        self, image_dir: str, dataset_file: str, transform=None
    ):
        """
        RavenRPM Dataset. Does not respect the train-test split used in the directory. Whill be split later using torch's random_split.

        Parameters
        ----------
        image_dir : str
            Path to RAVEN-F directory (including 'RAVEN-F')
        dataset_file : str
            Path to the dataset ``.csv`` file. If the file does not exist, it will be created.
        transform : Callable (Optional)
            Transformations to apply to the image
        """
        self.img_dir = image_dir

        if not os.path.exists(dataset_file):
            self.create_dataset_csv(dataset_file)

        self.data_paths = pd.read_csv(dataset_file)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> list[dict]:
        """
        Returns lists of dicts with the following keys:
        - `img` (Tensor) : The image of the panel.
        - `masks` (list[Tensor]) : A list of masks, one for each shape in the panel.
        - `locs` (list[ndarray]) : A list of location markers, one for each shape in the panel. Each is [center_x, center_y, height, width]
        - `type` (str) : One of `task[0-7]`, `target`, `distractor`.
        """
        npz_path = self.data_paths.iloc[idx, 0]
        xml_path = self.data_paths.iloc[idx, 1]

        ret = []

        # load files
        with np.load(npz_path) as f: # pyright: ignore[reportArgumentType]
            imgs = f["image"]
            target = f["target"]
        xml = get_xml(xml_path) # pyright: ignore[reportArgumentType]

        for id1 in range(len(imgs)):
            pan = {}
            img = imgs[id1]

            # Apply image transformation
            ogheight, ogwidth = img.shape
            img = np.tile(img, (3, 1, 1)).transpose([1, 2, 0])
            height, width = img.shape[:2]

            img = self.transform(img) if self.transform else img

            pan["img"] = img

            components = xml["Data"]["Panels"]["Panel"][id1]["Struct"]["Component"]
            components = components if isinstance(components, list) else [components]

            shape_masks = []
            shape_locs = []

            for comp in components:
                shapes = comp["Layout"]["Entity"]
                shapes = shapes if isinstance(shapes, list) else [shapes]

                for shape in shapes:
                    # location_information
                    real_bbox = ast.literal_eval(shape["@real_bbox"])
                    center_x = np.ceil(real_bbox[1] * width)
                    center_y = np.ceil(real_bbox[0] * height)
                    shape_width = real_bbox[3] * width
                    shape_height = real_bbox[2] * height
                    loc = [
                        max(np.floor(center_x - 0.5 * shape_width), 0),
                        max(np.floor(center_y - 0.5 * shape_height), 0),
                        np.ceil(center_x + 0.5 * shape_width),
                        np.ceil(center_y + 0.5 * shape_height),
                    ]
                    shape_locs.append(loc)

                    # mask
                    poly = [
                        (x + 0.5, y + 0.5) for x, y in ast.literal_eval(shape["@mask"])
                    ]
                    poly = [p for x in poly for p in x]
                    poly = [poly]
                    if len(poly) == 0:
                        mask = np.zeros((ogheight, ogwidth)).astype(bool)
                    else:
                        rles = mask_util.frPyObjects(poly, ogheight, ogwidth)
                        rle = mask_util.merge(rles)
                        mask = mask_util.decode(rle).astype(np.uint8)
                    mask = mask.astype(np.float32)
                    shape_masks.append(TF.to_tensor(mask).to(torch.float32))

            pan["masks"] = shape_masks
            pan["locs"] = shape_locs

            if id1 <= 7:
                pan["type"] = f"task{id1}"
            elif id1 - 8 == target:
                pan["type"] = "target"
            else:
                pan["type"] = "distractor"

            ret.append(pan)

        return ret

    def create_dataset_csv(self, dataset_file: str) -> None:
        """
        Called in the constructor, if no dataset .csv file is found. Creates a dataset .csv file.
        This stores the paths to the images and annotations to have them in a single accessible location.
        """
        log = logging.getLogger("reasoning_data")
        log.info("Creating RavenMatrices .csv file. This should only happen once.")

        npzs = []
        xmls = []

        files = Path(self.img_dir).rglob("*.xml")

        for file in files:
            xmls.append(file.as_posix())
            npzs.append(file.with_suffix(".npz").as_posix())

        df = pd.DataFrame({"npz": npzs, "xml": xmls})
        df.to_csv(dataset_file, index=False)
        log.debug("RavenTasks .csv file created.")


# --- Main, can be used for creating the dataset file ---

if __name__ == "__main__":
    log.init()

    with open(os.path.join(CWD, "global_cfg.yml"), "r") as f:
        global_cfg = CfgNode.load_cfg(f)

    RavenMatrices(
        image_dir = os.path.join(global_cfg.DATA.path, "RAVEN-F"),
        dataset_file = os.path.join(CWD, global_cfg.DATA.RavenMatrices.dataset_file),
    )

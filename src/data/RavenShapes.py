import ast
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import get_xml

from .RAVEN_FAIR.src.const import TYPE_VALUES as SHAPES


class RavenShapes(Dataset):
    """
    Torch Dataset Class for RAVEN-F (shapes only), meaning it produces panels with shape masks.
    """

    def __init__(
        self,
        image_dir: str,
        dataset_file: str,
        transform=None,
        mask_transform=None,
        target_transform=None,
        apply_mask=True,
    ) -> None:
        """
        RavenShapes Dataset. Does not respect the train-test split used in the directory. Whill be split later using torch's random_split.

        ### PARAMETERS
        - `image_dir` (str) : Directory containing the RAVEN(-F) dataset.
        - `dataset_file` (str) : Path to the dataset `.csv` file. If the file does not exist, it will be created.
        - `transform` (function) : Transformation to apply to the image.
        - `mask_transform` (function) : Transformation to apply to the mask. If `None`, `transform` will be used. If you need a transform for img but no transform for mask, use the identity: `lambda x: x`.
        - `target_transform` (function) : Transformation to apply to the target.
        """
        self.img_dir = image_dir
        if not os.path.exists(dataset_file):
            self.create_dataset_csv(dataset_file)

        self.img_labels = pd.read_csv(dataset_file)

        self.transform = transform
        self.mask_transform = transform if mask_transform is None else mask_transform
        self.target_transform = target_transform

        self.apply_mask = apply_mask

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item of index `idx` in the format of (image, mask), with both image and mask being of type `torch.Tensor`.
        """
        item_data = self.img_labels.iloc[idx]
        img_file = os.path.join(self.img_dir, item_data[0])
        ids = item_data[1].split("_")

        label = item_data[2]

        def get_id(id: int) -> int:
            return int(ids[id])

        # Load image
        with np.load(img_file) as npz:
            img_array = npz["image"][get_id(1)]
            img = np.tile(img_array, (3, 1, 1)).transpose([1, 2, 0])

        # get shape metadata
        metadata = get_xml(img_file[:-4] + ".xml")
        comps = metadata["Data"]["Panels"]["Panel"][get_id(1)]["Struct"]["Component"]
        comps = comps if type(comps) is list else [comps]
        ents = comps[get_id(2)]["Layout"]["Entity"]
        ents = ents if type(ents) is list else [ents]
        ent = ents[get_id(3)]

        # get poly mask
        poly = [(x + 0.5, y + 0.5) for x, y in ast.literal_eval(ent["@mask"])]
        poly = [p for x in poly for p in x]
        poly = [poly]

        # create mask based on detectron2's polygon_to_bitmask, but use only cocoapi for easier portability
        width, height = img.shape[:2]
        if len(poly) == 0:
            mask = np.zeros((height, width)).astype(bool)
        else:
            rles = mask_util.frPyObjects(poly, height, width)
            rle = mask_util.merge(rles)
            mask = mask_util.decode(rle).astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        if self.target_transform:
            label = self.target_transform(label)

        img = img.astype(np.float32) / 255
        mask = mask.astype(np.float32)

        if self.apply_mask:
            img = img + ((1 - mask).reshape(mask.shape[0], mask.shape[1], 1))
            img = img.clip(max=1)

        return (TF.to_tensor(img), TF.to_tensor(mask).to(torch.float32))

    def create_dataset_csv(self, dataset_file: str) -> bool:
        """
        Called in the constructor, if no dataset .csv file is found. Creates a dataset .csv file.
        The main loop follows closely the implementation of segmentation.dataset_registration.format_raven
        """
        log = logging.getLogger("CREATE RAVEN DATASET CSV")
        log.info("Creating RAVEN .csv file. This should only happen once.")

        img_file = []
        shape_id = []
        shape_type = []

        ##### uncomment for no overlap version, comment three lines after #####
        # ics = re.compile('in_center_single_out_center_single')
        # icd = re.compile('in_distribute_four_out_center_single')
        # files = Path(self.img_dir).rglob('*.xml*')
        # files = list(files)
        # kill = []
        # for idx, f in enumerate(files):
        #     if ics.match(f.parent.stem):
        #         kill.append(idx)
        #     if icd.match(f.parent.stem):
        #         kill.append(idx)
        # files = [f for idx, f in enumerate(files) if idx not in kill]
        # total_iterations = len(files)

        files = Path(self.img_dir).rglob("*.xml")
        total_iterations = len(list(files))
        files = Path(self.img_dir).rglob("*.xml")

        # Loop Level 1: Files
        for id1, l1 in tqdm(enumerate(files), total=total_iterations):
            # Load Metadata of file:
            file_string = l1.as_posix()
            file_dict = get_xml(file_string)
            corresp_data = file_string[:-4] + ".npz"

            # how many images are there (should usually be 16):
            with np.load(corresp_data) as npz:
                imgs = npz["image"]
            try:
                assert len(imgs) == 16, f"file {file_string} does not contain 16 images"
            except AssertionError as e:
                log.error(e)

            # Loop Level 2: Panels
            for id2 in range(len(imgs)):
                # Get image
                components = file_dict["Data"]["Panels"]["Panel"][id2]["Struct"][
                    "Component"
                ]
                components = components if type(components) is list else [components]

                # Loop Level 3: Components
                for id3, l3 in enumerate(
                    components
                ):  # using this id doesn't make any sense, but it helps for getting the shape out of the archive later.
                    shapes = l3["Layout"]["Entity"]
                    shapes = shapes if type(shapes) is list else [shapes]

                    # Loop Level 4: Shapes
                    for id4, l4 in enumerate(shapes):
                        type_id = int(l4["@Type"])

                        img_file.append(
                            corresp_data.replace(self.img_dir, "")
                        ) if self.img_dir[-1] == "/" else img_file.append(
                            corresp_data.replace(self.img_dir, "")[1:]
                        )
                        assert img_file[-1][0] != "/", (
                            f'Something went wrong... relative path to image has a leading "/": {img_file[-1]}'
                        )
                        shape_id.append(f"{id1}_{id2}_{id3}_{id4}")
                        shape_type.append(SHAPES[type_id])

        df = pd.DataFrame({
            "img_file": img_file,
            "shape_id": shape_id,
            "shape_type": shape_type,
        })
        df.to_csv(dataset_file, index=False)

        log.debug("create_dataset_csv complete")
        return True

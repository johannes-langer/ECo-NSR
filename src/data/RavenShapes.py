import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from yacs.config import CfgNode

CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())

from src.data.RAVEN_FAIR.src.const import TYPE_VALUES as SHAPES  # noqa: E402
from src.data.utils import convert_stringlist, get_xml  # noqa: E402


class RavenShapesDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for RAVEN-F.
    """

    def __init__(self, cfg: CfgNode, pin_memory : bool = False):
        super().__init__()

        self.cfg = cfg
        self.pin = pin_memory

        self.image_dir = os.path.join(cfg.DATA.path, "RAVEN-F")
        self.dataset_file = os.path.join(CWD, cfg.DATA.RavenShapes.dataset_file)
        self.transform = Compose([
            ToTensor(),
            Resize((
                cfg.REPRESENTATION.TRAINING.patch_size,
                cfg.REPRESENTATION.TRAINING.patch_size,
            )),
        ])

    def setup(self, *args, **kwargs) -> None:
        """
        Create dataset and splits.
        """
        self.ds = RavenShapes(self.image_dir, self.dataset_file, self.transform)
        self.train, self.val, self.test = random_split(
            self.ds, self.cfg.DATA.RavenShapes.split, torch.Generator().manual_seed(self.cfg.REPRESENTATION.TRAINING.seed)
        )

    def train_dataloader(self) -> DataLoader:
        """ """
        return DataLoader(
            self.train,
            batch_size = self.cfg.REPRESENTATION.TRAINING.batch_size,
            num_workers = self.cfg.REPRESENTATION.TRAINING.num_workers,
            pin_memory = self.pin,
            persistent_workers = True,
            shuffle = True,
        )

    def val_dataloader(self) -> DataLoader:
        """ """
        return DataLoader(
            self.val,
            batch_size = self.cfg.REPRESENTATION.TRAINING.batch_size,
            num_workers = self.cfg.REPRESENTATION.TRAINING.num_workers,
            pin_memory = self.pin,
            persistent_workers = True,
            shuffle = False,
        )

    def test_dataloader(self) -> DataLoader:
        """ """
        return DataLoader(
            self.test,
            batch_size = self.cfg.REPRESENTATION.TRAINING.batch_size,
            num_workers = self.cfg.REPRESENTATION.TRAINING.num_workers,
            pin_memory = self.pin,
            persistent_workers = True,
            shuffle = False,
        )

    def predict_dataloader(self) -> DataLoader:
        """ """
        return DataLoader(
            self.ds,
            batch_size = self.cfg.REPRESENTATION.TRAINING.batch_size,
            num_workers = self.cfg.REPRESENTATION.TRAINING.num_workers,
            pin_memory = self.pin,
            persistent_workers = True,
            shuffle = False,
        )


class RavenShapes(Dataset):
    """
    Torch Dataset Class for RAVEN-F (shapes only), meaning it produces single-shape crops of panels.
    This Dataset is used for representation training. Uses bounding box annotations.
    """

    def __init__(
        self,
        image_dir: str,
        dataset_file: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        RavenShapes Dataset. Does not respect the train-test split used in the directory. Whill be split later using torch's random_split.

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
        self.dataset_file = dataset_file
        if not os.path.exists(dataset_file):
            self.create_dataset_csv(dataset_file)

        self.img_labels = pd.read_csv(dataset_file)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, np.ndarray]:
        """
        Get bounding box crop of item `idx`.
        """
        item_data = self.img_labels.iloc[idx]
        img_file = os.path.join(self.img_dir, item_data.iloc[0])
        ids = item_data.iloc[1].split("_")

        def get_id(id: int) -> int:
            return int(ids[id])

        # Load image
        with np.load(img_file) as npz:
            img_array = npz["image"][get_id(1)]
            img = np.tile(img_array, (3, 1, 1)).transpose([1, 2, 0])

        height, width, _ = img.shape

        # get shape metadata
        metadata = get_xml(img_file[:-4] + ".xml")
        comps = metadata["Data"]["Panels"]["Panel"][get_id(1)]["Struct"]["Component"]
        comps = comps if type(comps) is list else [comps]
        ents = comps[get_id(2)]["Layout"]["Entity"]
        ents = ents if type(ents) is list else [ents]
        ent = ents[get_id(3)]

        real_bbox = convert_stringlist(ent["@real_bbox"])

        # calculate bounding box
        center_x = np.ceil(real_bbox[1] * width)
        center_y = np.ceil(real_bbox[0] * height)
        ent_width = real_bbox[3] * width
        ent_height = real_bbox[2] * height
        bbox = [
            max(np.floor(center_x - 0.5 * ent_width), 0),
            max(np.floor(center_y - 0.5 * ent_height), 0),
            np.ceil(center_x + 0.5 * ent_width),
            np.ceil(center_y + 0.5 * ent_height),
        ]

        # crop image
        img = img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2]), :]

        if img.shape[0] == 0 or img.shape[1] == 0:
            print(ids)

        if self.transform:
            img = self.transform(img)

        return img

    def create_dataset_csv(self, dataset_file: str) -> None:
        """
        Called in the constructor, if no dataset .csv file is found. Creates a dataset .csv file.
        The main loop follows closely the implementation of dataset_registration.format_raven

        Parameters
        ----------
        dataset_file : str
            Path to the to-be-created file
        """
        log = logging.getLogger("representation_data")
        log.info(
            "Creating .csv file for the representation torch.Dataset. This should only happen once."
        )

        img_file = []
        shape_id = []
        shape_type = []

        # exclude patterns which contain overlapping shapes
        ics = re.compile("in_center_single_out_center_single")
        icd = re.compile("in_distribute_four_out_center_single")
        files = Path(self.img_dir).rglob("*.xml*")
        files = list(files)
        kill = []
        for idx, f in enumerate(files):
            if ics.match(f.parent.stem):
                kill.append(idx)
            if icd.match(f.parent.stem):
                kill.append(idx)
        files = [f for idx, f in enumerate(files) if idx not in kill]
        total_iterations = len(files)

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
                        bbox = convert_stringlist(l4["@real_bbox"])
                        if bbox[2] <= 0.05 or bbox[3] <= 0.05:
                            continue  # skip entities that are too small as they probably aren't shapes.

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

# ECo-NSR
_Supplementary material for the paper "Enabling Neuro-Symbolic Reasoning by Making Cognitive Processes Explicit"._

---

## Setup

## Reproducing Our Results

## Using ECo-NSR with other datasets

At this stage, while ECo-NSR needs little conceptual adjustment for suitable datasets, the code is heavily tailored to the data format of the RAVEN-F dataset. While we cannot account for every possible existing (and future) dataset, here are some directions to help you adjust the code to your needs:

- `src/data/dataset_registration` is responsible for pre-processing the dataset for segmentation training. Replace `format_raven()` according to the guide in [detectron2's documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html) and register using `register_dataset_raw()`. Using datasets like this is slow, so use a snippet like `generate_coco_json()` in `src/data/generate_raven` to ONCE generate a coco json file, and use the appropriate registration fuction in the future.
- The code uses the custom `npz_mapper`. Adjust your dataloader to your data as described [here](https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html#write-a-custom-dataloader).

## Notes

- The `RAVEN_FAIR` submodule is a fork of the [original repository](https://github.com/yanivbenny/RAVEN_FAIR) which added support for Python3.

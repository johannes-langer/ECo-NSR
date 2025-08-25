# ECo-NSR
_Supplementary material for the paper "Enabling Neuro-Symbolic Reasoning by Making Cognitive Processes Explicit"._

---

## Setup

### Required

All experiments were run using Python 3.13.5 in a conda environment, but **Python>=3.9** should suffice. Only Popper requires **SWI-Prolog>=9.2** due to the `janus-swi` requirement. Popper 3.1 requires SWI-Prolog only at >=9.0.4. According to the release notes, the current version of Popper isn't necessarily more powerful, but significantly more efficient.

Install most requirements using

```bash
python -m pip install -r requirements.txt
```

Then, install detectron using

```bash
python -m pip install --no-build-isolation --use-pep517 'git+https://github.com/facebookresearch/detectron2.git'
```

**(Requires SWI-Prolog>=9.2)** Lastly, install Popper using

```bash
python -m pip install --use-pep517 git+https://github.com/logic-and-learning-lab/Popper@main
```

### Optional

- Install the NuWLS-c anytime solver for Popper ([installation instructions](https://github.com/logic-and-learning-lab/Popper/blob/main/solvers.md))

## Reproducing Our Results

We assume that all listed commands are run from the top directory and an environment where all required packages are installed. Data-generation and model training can all be completed in a single command using

```bash
python main.py
```

If you want to trace the steps in more detail or only run parts of them (or not run them all at once), the components offer hook ins to be called directly. Note that as most matrices cannot be solved by Popper, these files are more exploratory.

### **1. Prepare Data**

```bash
python src/data/generate_raven.py
```

### **2. Train Segmentation**

```bash
python src/segmentation/segment.py
```

Training can be resumed from the most recent checkpoint using the command line option `--resume`

### **3. Train Representation**

```bash
python src/representation/train_vae.py
```

Training can be resumed from a checkpoint using the command line option `--resume <path-to-.ckpt>`

### **4. Run Popper**

## Using ECo-NSR with other datasets

At this stage, while ECo-NSR needs little conceptual adjustment for suitable datasets, the code is heavily tailored to the data format of the RAVEN-F dataset. While we cannot account for every possible existing (and future) dataset, here are some directions to help you adjust the code to your needs:

- `src/data/dataset_registration` is responsible for pre-processing the dataset for segmentation training. Replace `format_raven()` according to the guide in [detectron2's documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html) and register using `register_dataset_raw()`. Using datasets like this is slow, so use a snippet like `generate_coco_json()` in `src/data/generate_raven` to ONCE generate a coco json file, and use the appropriate registration fuction in the future.
- The code uses the custom `npz_mapper`. Adjust your dataloader to your data as described [here](https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html#write-a-custom-dataloader).
- The `RavenShapes` dataset is used in VAE training, create your dataset in a similar fashion.
- The `RavenMatrices` dataset is used as the basis for reasoning. However, it is also used for the clustering, so those functions need to be adjusted to different layouts. The same is true for the functions in `src/reasoning/to_popper.py`
- `base_bias.pl` and `base_bk.pl` are also designed for RAVEN specifically and need to be adjusted.

## Notes

- The `RAVEN_FAIR` submodule is a fork of the [original repository](https://github.com/yanivbenny/RAVEN_FAIR) which added support for Python3.
- The `PyTorch_VAE` submodule is a fork of the [original repository](https://github.com/AntixK/PyTorch-VAE) which added support modern PyTorch and PyTorch Lightning.
- In the future the `--use-pep517` option for installing detectron2 and Popper may not be required, depending on package updates. Given project maintenance, we expect this to remain necessary for detectron2, with Popper most likely being updated at some point in the future.

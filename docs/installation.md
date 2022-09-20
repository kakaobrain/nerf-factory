---
layout: default
title: Installation 
nav_order: 2
---

# Installation
{: .no_toc }

How to install NeRF-Factory?
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Environment Setting

```bash
# Note that CUDA >= 11.0
conda create -n nerf_factory -c anaconda python=3.8
conda activate nerf_factory
conda install pytorch==1.11.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.3 \
-c pytorch -c conda-forge
pip3 install imageio tqdm requests configargparse scikit-image imageio-ffmpeg piqa \
pytorch_lightning==1.5.5 opencv-python gin-config gdown
```

## Preparing Dataset

We provide an automatic download script for all datasets. 

### NeRF-Blender Dataset

```bash
bash scripts/download_data.sh nerf_synthetic
```

### NeRF-LLFF Dataset
```bash
bash scripts/download_data.sh nerf_llff
```

### NeRF-360 Dataset
```bash
bash scripts/download_data.sh nerf_real_360
```

### Tanks and Temples Dataset
```bash
bash scripts/download_data.sh tanks_and_temples
```

### LF Dataset
```bash
bash scripts/download_data.sh lf
```

### NeRF-360-v2 Dataset
```bash
bash scripts/download_data.sh nerf_360_v2
```

### Ref-NeRF Shiny Blender Dataset
```bash
bash scripts/download_data.sh shiny_blender
```

## Run the Code!

A very simple script to run the code.


### Training Code

A script for running the training code.

```bash
python3 run.py --ginc configs/[model]/[data].gin --scene [scene]

## ex) run training nerf on chair scene of blender dataset
python3 run.py --ginc configs/nerf/blender.gin --scene chair
```

### Evaluation Code

A script for running the evaluation code only.

```bash
python3 run.py --ginc configs/[model]/[data].gin --scene [scene] \
--ginb run.run_train=False

## ex) run evaluating nerf on chair scene of blender dataset
python3 run.py --ginc configs/nerf/blender.gin --scene chair \
--ginb run.run_train=False
```


## Custom

How to add the custom dataset and the custom model in NeRF-Factory?

### Custom Dataset

- Add files of the custom dataset on ```./data/[custom_dataset]```.
- Implement a dataset loader code on ```./src/data/data_util/[custom_dataset].py```.
- Implement a custom dataset class ```LitData[custom_dataset]``` on ```./src/data/litdata.py```.
- Add option of selecting the custom dataset on the function ```def select_dataset()``` of ```./utils/select_option.py```.
- Add gin config file for each model as ```./configs/[model]/[custom_dataset].gin```.

### Custom Model

- Implement a custom model code on ```./src/model/[custom_model]/model.py```.
- Implement a custom model's helper code on ```./src/model/[custom_model]/helper.py```.
- [Optional] If you need more code files for the custom model, you can add them in ```./src/model/[custom_model]/```.
- Add option of selecting the custom model on the function ```def select_model()``` of ```./utils/select_option.py```.
- Add gin config file for each model as ```./configs/[custom_model]/[dataset].gin```.

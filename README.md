# NeRF-Factory: An awesome PyTorch NeRF collection

![logo](https://user-images.githubusercontent.com/33657821/191188990-d15744b5-c030-48ac-9669-2a0600bacdec.png)

[Project Page](https://kakaobrain.github.io/NeRF-Factory/) | [Checkpoints](https://huggingface.co/nrtf/nerf_factory)

Attention all NeRF researchers! We are here with a PyTorch-reimplemented large-scale NeRF library. Our library is easily extensible and usable.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33657821/191189332-5684634a-f21f-42ef-ade9-140010ffba4c.gif" alt="animated" width=300 height=300/>
  <img src="https://user-images.githubusercontent.com/33657821/191189091-cb6bce5f-814f-4c09-8da4-af36af8455c2.gif" alt="animated" height=300/>
</p>


This contains PyTorch-implementation of 7 popular NeRF models.
- NeRF: [[Project Page]](https://www.matthewtancik.com/nerf) [[Paper]](https://arxiv.org/abs/2003.08934) [[Code]](https://github.com/bmild/nerf)
- NeRF++: [[Paper]](http://arxiv.org/abs/2010.07492) [[Code]](https://github.com/Kai-46/nerfplusplus)
- DVGO: [[Project Page]](https://sunset1995.github.io/dvgo/) [[Paper-v1]](https://arxiv.org/abs/2111.11215) [[Paper-v2]](https://arxiv.org/abs/2206.05085) [[Code]](https://github.com/sunset1995/DirectVoxGO)
- Plenoxels: [[Project Page]](https://alexyu.net/plenoxels/) [[Paper]](https://arxiv.org/abs/2112.05131) [[Code]](https://github.com/sxyu/svox2)
- Mip-NeRF: [[Project Page]](https://jonbarron.info/mipnerf/) [[Paper]](https://arxiv.org/abs/2103.13415) [[Code]](https://github.com/google/mipnerf)
- Mip-NeRF360: [[Project Page]](https://jonbarron.info/mipnerf360/) [[Paper]](https://arxiv.org/abs/2111.12077) [[Code]](https://github.com/google-research/multinerf)
- Ref-NeRF: [[Project Page]](https://dorverbin.github.io/refnerf/) [[Paper]](https://arxiv.org/abs/2112.03907) [[Code]](https://github.com/google-research/multinerf)

and also 7 popular NeRF datasets.
- NeRF Blender: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- NeRF LLFF: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- Tanks and Temples: [link](https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view?usp=sharing)
- LF: [link](https://drive.google.com/file/d/1gsjDjkbTh4GAR9fFqlIDZ__qR9NYTURQ/view?usp=sharing)
- NeRF-360: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- NeRF-360-v2: [link](https://jonbarron.info/mipnerf360/)
- Shiny Blender: [link](https://dorverbin.github.io/refnerf/)

You only need to do for running the code is:

```bash
python3 -m run --ginc configs/[model]/[data].gin
# ex) python3 -m run --ginc configs/nerf/blender.gin
```

We also provide convenient visualizers for NeRF researchers.


## Contributor
This project is created and maintained by [Yoonwoo Jeong](https://github.com/jeongyw12382), [Seungjoo Shin](https://github.com/seungjooshin), and [Kibaek Park](https://github.com/parkkibaek).

## Requirements
```
conda create -n nerf_factory -c anaconda python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip3 install -r requirements.txt

## Optional(Plenoxel)
pip3 install .

## Or you could directly build from nerf_factory.yml
conda env create --file nerf_factory.yml
```

## Command

```bash
python3 -m run --ginc configs/[model]/[data].gin
# ex) python3 -m run --ginc configs/nerf/blender.gin
```

## Preparing Dataset

We provide an automatic download script for all datasets.

```bash
# NeRF-blender dataset
bash scripts/download_data.sh nerf_synthetic
# NeRF-LLFF(NeRF-Real) dataset
bash scripts/download_data.sh nerf_llff
# NeRF-360 dataset
bash scripts/download_data.sh nerf_real_360
# Tanks and Temples dataset
bash scripts/download_data.sh tanks_and_temples
# LF dataset
bash scripts/download_data.sh lf
# NeRF-360-v2 dataset
bash scripts/download_data.sh nerf_360_v2
# Shiny-blender dataset
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
- [Optional] If you need more code files for the custom model, you can add them in ```./src/model/[custom_model]/```.- Add option of selecting the custom model on the function ```def select_model()``` of ```./utils/select_option.py```.
- Add gin config file for each model as ```./configs/[custom_model]/[dataset].gin```.

### License

Copyright (c) 2022 POSTECH, KAIST, and Kakao Brain Corp. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (see [LICENSE](https://github.com/kakaobrain/NeRF-Factory/tree/main/LICENSE) for details)

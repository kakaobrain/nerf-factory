---
layout: default
title: Ref-NeRF
parent: Models
nav_order: 7
---

# Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields

[[Project Page]](https://dorverbin.github.io/refnerf/) [[Paper]](https://arxiv.org/abs/2112.03907) [[Code]](https://github.com/google-research/multinerf)

Authors 
- Jonathan T. Barron
- Ben Mildenhall
- Dor Verbin
- Pratul P. Srinivasan
- Peter Hedman

---

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---

## What's Ref-NeRF?

**Ref-NeRF**

Current NeRF-like methods fail to reconstruct the scenes, which contain glossy and reflective surfaces. To overcome these failure cases, Ref-NeRF proposes a new parameterization with reflected radiance producing spatially-varying material properties. They show that Ref-NeRF outperforms the accuracy of the normal map which synthesizes the realistic reflective surface like a chrome ball. With the re-parameterization and regularizer, Ref-NeRF renders the scene including specularities and reflections with the correct normal map crucial for generating a shiny material scene. 

![refnerf]({{site.baseurl}}/assets/images/models/refnerf.png)

## Scores


### Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| chair | 35.84 | 0.9842 | 0.0223 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_chair_220901) |
| drums | 25.52 | 0.9340 | 0.0735 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_drums_220901) |
| ficus | 31.32 | 0.9712 | 0.0407 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_ficus_220901) |
| hotdog | 36.54 | 0.9806 | 0.0347 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_hotdog_220901) |
| lego | 35.79 | 0.9795 | 0.0240 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_lego_220901) |
| materials | 35.71 | 0.9848 | 0.0280 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_materials_220901) |
| mic | 35.96 | 0.9912 | 0.0104 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_mic_220901) |
| ship | 29.51 | 0.8707 | 0.1643 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_blender_ship_220901) |

### Shiny Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| ball | 43.09 | 0.9933 | 0.0921 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_shiny_blender_ball_220901) |
| car | 30.70 | 0.9554 | 0.0454 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_shiny_blender_car_220901) |
| coffee | 32.27 | 0.9668 | 0.1242 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_shiny_blender_coffee_220901) |
| helmet | 29.66 | 0.9585 | 0.1033 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_shiny_blender_helmet_220901) |
| teapot | 45.20 | 0.9960 | 0.0176 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_shiny_blender_teapot_220901) |
| toaster | 24.88 | 0.9141 | 0.1318 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/refnerf_shiny_blender_toaster_220901) |

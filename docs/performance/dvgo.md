---
layout: default
title: DVGO
parent: Models
nav_order: 3
---

# DVGO: Direct Voxel Grid Optimization

[[Project Page]](https://sunset1995.github.io/dvgo/) [[Paper-v1]](https://arxiv.org/abs/2111.11215) [[Paper-v2]](https://arxiv.org/abs/2206.05085) [[Code]](https://github.com/sunset1995/DirectVoxGO)

Authors 
- Cheng Sun
- Min Sun
- Hwann-Tzong Chen

---

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---

## What's DVGO?

**DVGO (Direct Voxel Grid Optimization)**

DVGO (Direct Voxel Grid Optimization) is introduced in the CVPR 2022 oral paper, “Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction”. NeRF trains the neural network from many hours to days for a single scene. DVGO proposes the fast convergence of the neural radiance fields and high-quality generation. By utilizing the coordinate-based neural network, two voxel grid representation learns geometry and view-dependent features of the 3D scene. DVGO proposed the post-activation interpolation on voxel, which predict high-quality synthesis around the sharp surface, and they regularize the voxel grid representation with several priors. It speeds up 2-3x faster than the original NeRF using shallow MLP, and reduces the training time from many hours to 15 minutes. 

![dvgo]({{site.baseurl}}/assets/images/models/dvgo.png)

## Scores


### Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| chair | 35.20 | 0.9809 | 0.0283 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_chair_220901) |
| drums | 25.53 | 0.9318 | 0.0795 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_drums_220901) |
| ficus | 33.23 | 0.9798 | 0.0258 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_ficus_220901) |
| hotdog | 37.44 | 0.9814 | 0.0345 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_hotdog_220901) |
| lego | 35.80 | 0.9781 | 0.0265 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_lego_220901) |
| materials | 30.58 | 0.9582 | 0.0526 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_materials_220901) |
| mic | 36.41 | 0.9909 | 0.0126 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_mic_220901) |
| ship | 30.52 | 0.8800 | 0.1500 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_ship_220901) |

### MS-Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint | 
|:---|:---:|:---:|:---:|:---:|
| chair | 37.36 | 0.9878 | 0.0161 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_chair_220901) |
| drums | 27.12 | 0.9458 | 0.0561 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_drums_220901) |
| ficus | 33.00 | 0.9831 | 0.0185 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_ficus_220901) |
| hotdog | 39.36 | 0.9879 | 0.0178 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_hotdog_220901) |
| lego | 35.71 | 0.9840 | 0.0174 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_lego_220901) |
| materials | 32.63 | 0.9767 | 0.0260 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_materials_220901) |
| mic | 37.93 | 0.9926 | 0.0101 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_mic_220901) |
| ship | 33.24 | 0.9235 | 0.0818 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_blender_multiscale_ship_220901) |

### LLFF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| fern | 24.92 | 0.7957 | 0.2793 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_fern_220901) |
| flower | 27.80 | 0.8430 | 0.1937 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_flower_220901) |
| fortress | 31.73 | 0.8933 | 0.1492 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_fortress_220901) |
| horns | 27.79 | 0.8571 | 0.2236 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_horns_220901) |
| leaves | 20.94 | 0.7038 | 0.3060 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_leaves_220901) |
| orchids | 20.27 | 0.6402 | 0.3204 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_orchids_220901) |
| room | 33.24 | 0.9552 | 0.1533 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_room_220901) |
| trex | 27.69 | 0.9025 | 0.2220 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_llff_trex_220901) |

### Tanks and Temples

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| M60 | 18.41 | 0.6435 | 0.4935 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_tanks_and_temples_tat_intermediate_M60_220901) |
| Playground | 21.83 | 0.6641 | 0.4948 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_tanks_and_temples_tat_intermediate_Playground_220901) |
| Train | 17.87 | 0.5749 | 0.4964 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_tanks_and_temples_tat_intermediate_Train_220901) |
| Truck | 21.71 | 0.6903 | 0.4562 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_tanks_and_temples_tat_training_Truck_220901) |

### LF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| Africa | 28.65 | 0.8676 | 0.3050 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_lf_africa_220901) |
| Basket | 21.98 | 0.8158 | 0.4056 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_lf_basket_220901) |
| Ship | 26.43 | 0.7968 | 0.3650 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_lf_ship_220901) |
| Statue | 29.86 | 0.8818 | 0.2973 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_lf_statue_220901) |
| Torch | 23.29 | 0.7513 | 0.4012 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_lf_torch_220901) |


### NeRF-360-v2

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| bicycle | 21.72 | 0.4246 | 0.5880 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_nerf_360_v2_bicycle_220901) |
| bonsai | 29.12 | 0.8443 | 0.3164 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_nerf_360_v2_bonsai_220901) |
| garden | 23.71 | 0.5544 | 0.4334 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_nerf_360_v2_garden_220901) |
| kitchen | 27.98 | 0.8025 | 0.2570 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_nerf_360_v2_kitchen_220901) |
| room | 30.23 | 0.8557 | 0.3342 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_nerf_360_v2_rpoom_220901) |
| stump | 22.74 | 0.4722 | 0.5716 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_nerf_360_v2_stump_220901) |


### Shiny Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| ball | 26.13 | 0.9007 | 0.2525 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_shiny_blender_ball_220901) |
| car | 26.90 | 0.9164 | 0.0806 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_shiny_blender_car_220901) |
| coffee | 31.48 | 0.9624 | 0.1414 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_shiny_blender_coffee_220901) |
| helmet | 27.75 | 0.9315 | 0.1541 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_shiny_blender_helmet_220901) |
| teapot | 44.79 | 0.9955 | 0.0186 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_shiny_blender_teapot_220901) |
| toaster | 22.18 | 0.8481 | 0.2196 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/dvgo_shiny_blender_toaster_220901) |

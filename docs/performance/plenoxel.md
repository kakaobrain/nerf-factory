---
layout: default
title: Plenoxels
parent: Models
nav_order: 4
---

# Plenoxels: Radiance Fields without Neural Networks

[[Project Page]](https://alexyu.net/plenoxels/) [[Paper]](https://arxiv.org/abs/2112.05131) [[Code]](https://github.com/sxyu/svox2)

Authors 
- Alex Yu
- Sara Fridovich-Keil
- Matthew Tancik
- Qinhong Chen
- Benjamin Recht
- Angjoo Kanazawa

---

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---

## What's Plenoxels?

**Plenoxels**

Plenoxels exploits the spherical harmonics coefficient to represent the sparse grid voxel representation. From the optimized spherical harmonics (SH) coefficient, Plenoxels synthesizes the view-dependent color image without any neural networks. Specifically, it interpolates the sampled grid SH coefficient which is close to the novel viewpoint ray input. Through the radiance fields without neural networks, Plenoxels renders the scene 11 minutes (1 day for Vanilla NeRF which is more than a 100× speedup)

![plenoxels]({{site.baseurl}}/assets/images/models/plenoxels.png)

## Scores

### Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| chair | 34.11 | 0.9769 | 0.0296 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_chair_220901) |
| drums | 25.42 | 0.9323 | 0.0657 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_drums_220901) |
| ficus | 31.94 | 0.9759 | 0.0258 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_ficus_220901) |
| hotdog | 36.69 | 0.9805 | 0.0367 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_hotdog_220901) |
| lego | 34.34 | 0.9753 | 0.0272 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_lego_220901) |
| materials | 29.26 | 0.9495 | 0.0553 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_materials_220901) |
| mic | 33.36 | 0.9851 | 0.0148 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_mic_220901) |
| ship | 30.10 | 0.8901 | 0.1313 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_ship_220901) |

### MS-Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| chair | 32.87 | 0.9688 | 0.0391 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_chair_220901) |
| drums | 25.34 | 0.9300 | 0.0688 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_drums_220901) |
| ficus | 30.35 | 0.9722 | 0.0311 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_ficus_220901) |
| hotdog | 34.89 | 0.9772 | 0.0363 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_hotdog_220901) |
| lego | 31.42 | 0.9653 | 0.0374 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_lego_220901) |
| materials | 28.48 | 0.9600 | 0.0537 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_materials_220901) |
| mic | 31.59 | 0.9791 | 0.0355 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_mic_220901) |
| ship | 29.04 | 0.8967 | 0.1009 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_blender_multiscale_ship_220901) |

### LLFF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| fern | 24.91 | 0.8076 | 0.2438 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_fern_220901) |
| flower | 28.17 | 0.8676 | 0.1733 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_flower_220901) |
| fortress | 31.36 | 0.8902 | 0.1661 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_fortress_220901) |
| horns | 27.77 | 0.8618 | 0.2218 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_horns_220901) |
| leaves | 21.19 | 0.7429 | 0.2165 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_leaves_220901) |
| orchids | 20.01 | 0.6527 | 0.2892 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_orchids_220901) |
| room | 31.33 | 0.9414 | 0.1884 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_room_220901) |
| trex | 26.54 | 0.8874 | 0.2523 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_llff_trex_220901) |

### Tanks and Temples

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| M60 | 17.74 | 0.6701 | 0.4525 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_tanks_and_temples_tat_intermediate_M60_220901) |
| Playground | 22.62 | 0.6891 | 0.4636 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_tanks_and_temples_tat_intermediate_Playground_220901) |
| Train | 17.66 | 0.6118 | 0.4608 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_tanks_and_temples_tat_intermediate_Train_220901) |
| Truck | 22.52 | 0.7373 | 0.3935 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_tanks_and_temples_tat_training_Truck_220901) |

### LF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| Africa | 14.99 | 0.6437 | 0.4713 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_lf_africa_220901) |
| Basket | 16.02 | 0.6935 | 0.4555 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_lf_basket_220901) |
| Ship | 25.40 | 0.8063 | 0.2886 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_lf_ship_220901) |
| Statue | 28.96 | 0.8900 | 0.2219 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_lf_statue_220901) |
| Torch | 24.84 | 0.8244 | 0.2757 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_lf_torch_220901) |

### NeRF-360-v2

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| bicycle | 21.42 | 0.4255 | 0.5709 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_nerf_360_v2_bicycle_220901) |
| bonsai | 26.21 | 0.8019 | 0.3305 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_nerf_360_v2_bonsai_220901) |
| counter | 25.62 | 0.7590 | 0.3520 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_nerf_360_v2_counter_220901) |
| garden | 23.13 | 0.5471 | 0.4360 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_nerf_360_v2_garden_220901) |
| kitchen | 25.09 | 0.6670 | 0.3560 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_nerf_360_v2_kitchen_220901) |
| room | 28.35 | 0.8373 | 0.3358 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_nerf_360_v2_room_220901) |
| stump | 22.88 | 0.5049 | 0.5381 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_nerf_360_v2_stump_220901) |


### Shiny Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| ball | 24.52 | 0.8316 | 0.2738 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_shiny_blender_ball_220901) |
| car | 26.11 | 0.9053 | 0.0855 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_shiny_blender_car_220901) |
| coffee | 31.55 | 0.9629 | 0.1442 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_shiny_blender_coffee_220901) |
| helmet | 26.94 | 0.9128 | 0.1704 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_shiny_blender_helmet_220901) |
| teapot | 44.25 | 0.9958 | 0.0158 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_shiny_blender_teapot_220901) |
| toaster | 19.50 | 0.7717 | 0.2443 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/plenoxel_shiny_blender_toaster_220901) |

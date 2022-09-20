---
layout: default
title: Mip-NeRF 360
parent: Models
nav_order: 6
---

# Mip-NeRF360: Unbounded Anti-Aliased Neural Radiance Fields

[[Project Page]](https://jonbarron.info/mipnerf360/) [[Paper]](https://arxiv.org/abs/2111.12077) [[Code]](https://github.com/google-research/multinerf)

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

## What's Mip-NeRF 360?

**Mip-NeRF 360**

As NeRF++ tackles the unbounded scene problem, they extend the Mip-NeRF to the unbounded scene to deal with the real-world arbitrary camera pose orientation. Proposal-based distillation method efficiently generates volumetric density. Volume representation with contracted Gaussian region accurately models the unbounded 360 scenes. A novel distortion-based regularizer removes the “floater” artifacts.

![mipnerf360]({{site.baseurl}}/assets/images/models/mipnerf360.png)

## Scores


### LLFF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| fern | 24.58 | 0.8047 | 0.2378 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_fern_220901) |
| flower | 27.81 | 0.8538 | 0.1607 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_flower_220901) |
| fortress | 31.17 | 0.8950 | 0.1259 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_fortress_220901) |
| horns | 28.03 | 0.8922 | 0.1592 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_horns_220901) |
| leaves | 20.28 | 0.7079 | 0.2989 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_leaves_220901) |
| orchids | 19.74 | 0.6488 | 0.2868 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_orchids_220901) |
| room | 33.55 | 0.9637 | 0.1209 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_room_220901) |
| trex | 27.87 | 0.9136 | 0.1827 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_llff_trex_220901) |

### Tanks and Temples

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| M60 | 20.09 | 0.7123 | 0.3991 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_tanks_and_temples_tat_intermediate_M60_220901) |
| Playground | 24.27 | 0.7401 | 0.3551 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_tanks_and_temples_tat_intermediate_Playground_220901) |
| Train | 19.74 | 0.6705 | 0.3813 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_tanks_and_temples_tat_intermediate_Train_220901) |
| Truck | 24.14 | 0.8004 | 0.2963 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_tanks_and_temples_tat_training_Truck_220901) |

### LF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| Africa | 29.58 | 0.9168 | 0.1984 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_lf_africa_220901) |
| Basket | 21.19 | 0.8714 | 0.2943 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_lf_basket_220901) |
| Ship | 30.16 | 0.8992 | 0.2329 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_lf_ship_220901) |
| Statue | 34.90 | 0.9521 | 0.1505 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_lf_statue_220901) |
| Torch | 25.86 | 0.8571 | 0.2707 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_lf_torch_220901) |

### 360-v2

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| bicycle | 22.86 | 0.5434 | 0.4553 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_nerf_360_v2_bicycle_220901) |
| bonsai | 32.97 | 0.9375 | 0.1439 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_nerf_360_v2_bonsai_220901) |
| counter | 29.29 | 0.8709 | 0.2118 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_nerf_360_v2_counter_220901) |
| garden | 26.01 | 0.7426 | 0.2703 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_nerf_360_v2_garden_220901) |
| kitchen | 31.99 | 0.9247 | 0.1090 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_nerf_360_v2_kitchen_220901) |
| room | 32.68 | 0.9258 | 0.1797 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_nerf_360_v2_room_220901) |
| stump | 25.28 | 0.6522 | 0.4153 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/mipnerf360_nerf_360_v2_stump_220901) |

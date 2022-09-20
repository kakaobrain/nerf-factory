---
layout: default
title: NeRF++
parent: Models
nav_order: 2
---

# NeRF++: Analyzing and Improving Neural Radiance Fields

[[Paper]](http://arxiv.org/abs/2010.07492) [[Code]](https://github.com/Kai-46/nerfplusplus)

Authors 
- Kai Zhang
- Gernot Riegler
- Noah Snavely
- Vladlen Koltun

---


## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---

## What's NeRF++?

**NeRF++**

Since vanilla NeRF suffers from the unbounded scene with implicit representation, NeRF++ proposes the inverted sphere parameterization, in which rays from the novel viewpoint are consistent with foreground and background. The dual volume representation calculates view-dependent color using the volumetric rendering with inner and outside sphere, which improve the quality of the unbounded 3D background model. NeRF++ extends the challenging scene including 360 captures within object-centric unbounded environments.

![nerfpp]({{site.baseurl}}/assets/images/models/nerfpp.png)

## Scores

### Tanks and Temples

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| M60 | 17.96 | 0.6359 | 0.4843 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_tanks_and_temples_tat_intermediate_M60_220901) |
| Playground | 22.91 | 0.7060 | 0.4333 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_tanks_and_temples_tat_intermediate_Playground_220901) |
| Train | 18.19 | 0.5810 | 0.4895 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_tanks_and_temples_tat_intermediate_Train_220901) |
| Truck | 22.60 | 0.7280 | 0.3999 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_tanks_and_temples_tat_training_Truck_220901) |

### LF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| Africa | 30.52 | 0.9021 | 0.2641 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_lf_africa_220901) |
| Basket | 22.13 | 0.8709 | 0.3178 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_lf_basket_220901) |
| Ship | 28.05 | 0.8470 | 0.3209 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_lf_ship_220901) |
| Statue | 32.49 | 0.9238 | 0.2431 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_lf_statue_220901) |
| Torch | 25.86 | 0.8395 | 0.3122 |[link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_lf_torch_220901) |

### NeRF-360-v2

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| bicycle | 21.43 | 0.4156 | 0.5822 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_nerf_360_v2_bicycle_220901) |
| bonsai | 31.67 | 0.8987 | 0.2413 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_nerf_360_v2_bonsai_220901) |
| counter | 27.72 | 0.8017 | 0.3056 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_nerf_360_v2_counter_220901) |
| garden | 24.80 | 0.6464 | 0.3568 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_nerf_360_v2_garden_220901) |
| kitchen | 29.47 | 0.8609 | 0.1799 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_nerf_360_v2_kitchen_220901) |
| room | 30.62 | 0.8732 | 0.2987 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_nerf_360_v2_room_220901) |
| stump | 24.77 | 0.6010 | 0.4420 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerfpp_nerf_360_v2_stump_220901) |
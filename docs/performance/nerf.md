---
layout: default
title: NeRF
parent: Models
nav_order: 1
---

# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

[[Project Page]](https://www.matthewtancik.com/nerf) [[Paper]](https://arxiv.org/abs/2003.08934) [[Code]](https://github.com/bmild/nerf)

Authors 
- Ben Mildenhall
- Pratul P. Srinivasan
- Matthew Tancik 
- Jonathan T. Barron
- Ravi Ramamoorthi
- Ren Ng

---

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---

## What's NeRF? 

**NeRF (Neural Radiance Fields)**

Neural Radiance Fields (NeRF) is proposed in the ECCV 2020 paper, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis” (an honorable mention for Best Paper). NeRF synthesizes the novel viewpoint of the scene, exploiting a collection of posed images (e.g. images with corresponding camera position and rotation). In the rendering step, NeRF synthesizes the RGB color of each novel viewpoint in terms of the appearance of the scene varied from the viewpoints.

![nerf]({{site.baseurl}}/assets/images/models/nerf.png)

NeRF model consists of small MLPs that learn the radiance distributions from each viewpoint. It uses the conventional volumetric rendering in training steps, minimizing the residual between synthesized and ground truth observed RGB value. Specifically, NeRF learns from the input with 3D location (x,y,z) and view direction(theta, phi) calculated from the posed images and ray sampling points, and MLP produces the emitted color and volume density (sigma).

From the effect of synthesizing the novel-view images, NeRF tremendously compresses the scene of the 3D geometry and appearance because the user can generate the novel-view scene, which is not possible in conventional video clips (collection of images). In short, around 30MB of NeRF weights generate the infinite image collection of 3D scenes, which has benefits for immersive experience VR and AR industry.

## Scores

### Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| chair | 34.93 | 0.9794 | 0.0293 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_chair_220901) |
| drums | 25.28 | 0.9292 | 0.0802 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_drums_220901) |
| ficus | 31.28 | 0.9718 | 0.0329 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_ficus_220901) |
| hotdog | 37.16 | 0.9803 | 0.0356 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_hotdog_220901) |
| lego | 34.38 | 0.9731 | 0.0321 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_lego_220901) |
| materials | 30.45 | 0.9558 | 0.0555 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_materials_220901) |
| mic | 35.18 | 0.9887 | 0.0146 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_mic_220901) |
| ship | 29.95 | 0.8784 | 0.1613 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_ship_220901) |

### MS-Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| chair | 32.83 | 0.9685 | 0.0394 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_chair_220901) |
| drums | 25.24 | 0.9278 | 0.0760 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_drums_220901) |
| ficus | 30.23 | 0.9715 | 0.0311 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_ficus_220901) |
| hotdog | 35.24 | 0.9791 | 0.0319 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_hotdog_220901) |
| lego | 31.45 | 0.9649 | 0.0381 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_lego_220901) |
| materials | 29.54 | 0.9661 | 0.0489 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_materials_220901) |
| mic | 32.20 | 0.9804 | 0.0346 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_mic_220901) |
| ship | 29.41 | 0.9016 | 0.1096 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_blender_multiscale_ship_220901) |

### LLFF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| fern | 25.19 | 0.8045 | 0.2597 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_fern_220901) |
| flower | 27.94 | 0.8467 | 0.1898 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_flower_220901) |
| fortress | 31.73 | 0.8950 | 0.1441 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_fortress_220901) |
| horns | 28.03 | 0.8585 | 0.2207 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_horns_220901) |
| leaves | 21.17 | 0.7141 | 0.2892 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_leaves_220901) |
| orchids | 20.29 | 0.6408 | 0.3210 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_orchids_220901) |
| room | 32.96 | 0.9542 | 0.1598 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_room_220901) |
| trex | 27.52 | 0.9009 | 0.2233 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_llff_trex_220901) |

### Tanks and Temples

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| M60 | 18.27 | 0.6447 | 0.4851 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_tanks_and_temples_tat_intermediate_M60_220901) |
| Playground | 21.68 | 0.6702 | 0.4945 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_tanks_and_temples_tat_intermediate_Playground_220901) |
| Train | 17.37 | 0.5581 | 0.5059 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_tanks_and_temples_tat_intermediate_Train_220901) |
| Truck | 21.44 | 0.6954 | 0.4475 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_tanks_and_temples_tat_training_Truck_220901) |

### LF

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| Africa | 28.53 | 0.8685 | 0.3062 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_lf_africa_220901) |
| Basket | 21.64 | 0.8146 | 0.3981 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_lf_basket_220901) |
| Ship | 26.26 | 0.7910 | 0.3646 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_lf_ship_220901) |
| Statue | 29.76 | 0.8745 | 0.3059 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_lf_statue_220901) |
| Torch | 23.24 | 0.7542 | 0.3947 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_lf_torch_220901) |

### NeRF-360-v2

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| bicycle | 21.82 | 0.4306 | 0.5732 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_nerf_360_v2_bicycle_220901) |
| bonsai | 29.03 | 0.8328 | 0.3242 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_nerf_360_v2_bonsai_220901) |
| counter | 26.98 | 0.7708 | 0.3574 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_nerf_360_v2_counter_220901) |
| garden | 23.64 | 0.5642 | 0.4292 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_nerf_360_v2_garden_220901) |
| kitchen | 27.16 | 0.7367 | 0.3165 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_nerf_360_v2_kitchen_220901) |
| room | 30.10 | 0.8603 | 0.3223 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_nerf_360_v2_room_220901) |
| stump | 22.93 | 0.4811 | 0.5531 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_nerf_360_v2_stump_220901) |

### Shiny Blender

| Scene | PSNR | SSIM | LPIPS | Checkpoint |
|:---|:---:|:---:|:---:|:---:|
| ball | 27.18 | 0.9356 | 0.2088 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_shiny_blender_ball_220901) |
| car | 26.42 | 0.9197 | 0.0710 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_shiny_blender_car_220901) |
| coffee | 30.64 | 0.9641 | 0.1330 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_shiny_blender_coffee_220901) |
| helmet | 27.61 | 0.9388 | 0.1436 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_shiny_blender_helmet_220901) |
| teapot | 45.37 | 0.9963 | 0.0133 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_shiny_blender_teapot_220901) |
| toaster | 22.51 | 0.8856 | 0.1407 | [link](https://huggingface.co/nrtf/nerf_factory/tree/main/nerf_shiny_blender_toaster_220901) |


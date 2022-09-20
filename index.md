---
layout: default
title: Home
nav_order: 1
description: "This"
permalink: /
---

# **NeRF-Factory**
{: .fs-9 }

An awesome NeRF collection.
{: .fs-6 .fw-300 }

[Installation](./docs/installation){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 } [View it on GitHub](https://github.com/kakaobrain/NeRF-Factory/){: .btn .fs-5 .mb-4 .mb-md-0 }

---

![logo](https://user-images.githubusercontent.com/33657821/191188990-d15744b5-c030-48ac-9669-2a0600bacdec.png)


## Motivation

Neural fields has recently been one of the most discussed topics in computer vision. As seen by a large number of NeRF papers, we can realize that it has been developed rapidly. However, we have not found a large-scale library that collects multiple NeRF implementations into an integrated form. Thus, this library, called **NeRF-Factory**, provides a convenient tool for evaluating and comparing NeRF models. Our library is super-easy to add your custom data and model by integrating format of codes.


## Supports

- Allows multi-GPU training with PyTorch-Lightning except for models that are even inefficient when running with Multi-GPU, such as Plenoxels and DVGO.
- A project page that includes instructions, results, and links for the pretrained models. 
- Dividing the NeRF’s training process into three phases: ray generation, network forwarding, and optimization. 
- Convenient switching of respective process to desired options by simply switching the config.
- Interactive visualization of rendered images for convenient comparison between trained models. 


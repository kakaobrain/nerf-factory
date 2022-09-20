---
layout: default
title: Tanks and Temples
parent: Visualization
nav_order: 4
---

<script defer src="https://unpkg.com/img-comparison-slider@7/dist/index.js"></script>
<link rel="stylesheet" href="https://unpkg.com/img-comparison-slider@7/dist/styles.css"/>
<script src="{{site.baseurl}}/assets/js/my-js.js"></script>
<link rel="stylesheet" href="{{site.baseurl}}/assets/css/my-css.css"/>

---

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---

## M60

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_tanks_and_temples_tat_intermediate_M60" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_tanks_and_temples_tat_intermediate_M60_220901/render_model/image000.jpg">
      <figcaption id="left_caption_tanks_and_temples_tat_intermediate_M60" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 18.27
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_tanks_and_temples_tat_intermediate_M60" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_tanks_and_temples_tat_intermediate_M60_220901/render_model/image000.jpg">
      <figcaption id="right_caption_tanks_and_temples_tat_intermediate_M60" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 20.09
      </figcaption>
    </figure>
  </img-comparison-slider>
  <table width="100%">
    <tr>
      <td align="center">
        Left Model
      </td>
      <td align="center">
        Frame
      </td>
      <td align="center">
        Right Model
      </td>
    </tr>
    <tr>
      <td align="center">
        <select id="left_select_tanks_and_temples_tat_intermediate_M60" onchange="select_model(this, 'left', 'tanks_and_temples', 'tat_intermediate_M60')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_tanks_and_temples_tat_intermediate_M60" type="range" min="0" max="35" value="0" onchange="select_frame(this, 'tanks_and_temples', 'tat_intermediate_M60')">
      </td>
      <td align="center">
        <select id="right_select_tanks_and_temples_tat_intermediate_M60" onchange="select_model(this, 'right', 'tanks_and_temples', 'tat_intermediate_M60')">
          <option value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## Playground

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_tanks_and_temples_tat_intermediate_Playground" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_tanks_and_temples_tat_intermediate_Playground_220901/render_model/image000.jpg">
      <figcaption id="left_caption_tanks_and_temples_tat_intermediate_Playground" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 21.68
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_tanks_and_temples_tat_intermediate_Playground" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_tanks_and_temples_tat_intermediate_Playground_220901/render_model/image000.jpg">
      <figcaption id="right_caption_tanks_and_temples_tat_intermediate_Playground" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 24.27
      </figcaption>
    </figure>
  </img-comparison-slider>
  <table width="100%">
    <tr>
      <td align="center">
        Left Model
      </td>
      <td align="center">
        Frame
      </td>
      <td align="center">
        Right Model
      </td>
    </tr>
    <tr>
      <td align="center">
        <select id="left_select_tanks_and_temples_tat_intermediate_Playground" onchange="select_model(this, 'left', 'tanks_and_temples', 'tat_intermediate_Playground')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_tanks_and_temples_tat_intermediate_Playground" type="range" min="0" max="31" value="0" onchange="select_frame(this, 'tanks_and_temples', 'tat_intermediate_Playground')">
      </td>
      <td align="center">
        <select id="right_select_tanks_and_temples_tat_intermediate_Playground" onchange="select_model(this, 'right', 'tanks_and_temples', 'tat_intermediate_Playground')">
          <option value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## Train

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_tanks_and_temples_tat_intermediate_Train" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_tanks_and_temples_tat_intermediate_Train_220901/render_model/image000.jpg">
      <figcaption id="left_caption_tanks_and_temples_tat_intermediate_Train" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 17.37
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_tanks_and_temples_tat_intermediate_Train" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_tanks_and_temples_tat_intermediate_Train_220901/render_model/image000.jpg">
      <figcaption id="right_caption_tanks_and_temples_tat_intermediate_Train" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 19.74
      </figcaption>
    </figure>
  </img-comparison-slider>
  <table width="100%">
    <tr>
      <td align="center">
        Left Model
      </td>
      <td align="center">
        Frame
      </td>
      <td align="center">
        Right Model
      </td>
    </tr>
    <tr>
      <td align="center">
        <select id="left_select_tanks_and_temples_tat_intermediate_Train" onchange="select_model(this, 'left', 'tanks_and_temples', 'tat_intermediate_Train')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_tanks_and_temples_tat_intermediate_Train" type="range" min="0" max="42" value="0" onchange="select_frame(this, 'tanks_and_temples', 'tat_intermediate_Train')">
      </td>
      <td align="center">
        <select id="right_select_tanks_and_temples_tat_intermediate_Train" onchange="select_model(this, 'right', 'tanks_and_temples', 'tat_intermediate_Train')">
          <option value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## Truck

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_tanks_and_temples_tat_training_Truck" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_tanks_and_temples_tat_training_Truck_220901/render_model/image000.jpg">
      <figcaption id="left_caption_tanks_and_temples_tat_training_Truck" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 21.44
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_tanks_and_temples_tat_training_Truck" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_tanks_and_temples_tat_training_Truck_220901/render_model/image000.jpg">
      <figcaption id="right_caption_tanks_and_temples_tat_training_Truck" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 24.14
      </figcaption>
    </figure>
  </img-comparison-slider>
  <table width="100%">
    <tr>
      <td align="center">
        Left Model
      </td>
      <td align="center">
        Frame
      </td>
      <td align="center">
        Right Model
      </td>
    </tr>
    <tr>
      <td align="center">
        <select id="left_select_tanks_and_temples_tat_training_Truck" onchange="select_model(this, 'left', 'tanks_and_temples', 'tat_training_Truck')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_tanks_and_temples_tat_training_Truck" type="range" min="0" max="24" value="0" onchange="select_frame(this, 'tanks_and_temples', 'tat_training_Truck')">
      </td>
      <td align="center">
        <select id="right_select_tanks_and_temples_tat_training_Truck" onchange="select_model(this, 'right', 'tanks_and_temples', 'tat_training_Truck')">
          <option value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

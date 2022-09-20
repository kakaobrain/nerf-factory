---
layout: default
title: LF
parent: Visualization
nav_order: 5
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

## Africa

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_lf_africa" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_lf_africa_220901/render_model/image000.jpg">
      <figcaption id="left_caption_lf_africa" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 28.53
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_lf_africa" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_lf_africa_220901/render_model/image000.jpg">
      <figcaption id="right_caption_lf_africa" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 29.58
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
        <select id="left_select_lf_africa" onchange="select_model(this, 'left', 'lf', 'africa')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_lf_africa" type="range" min="0" max="62" value="0" onchange="select_frame(this, 'lf', 'africa')">
      </td>
      <td align="center">
        <select id="right_select_lf_africa" onchange="select_model(this, 'right', 'lf', 'africa')">
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

## Basket

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_lf_basket" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_lf_basket_220901/render_model/image000.jpg">
      <figcaption id="left_caption_lf_basket" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 21.64
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_lf_basket" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_lf_basket_220901/render_model/image000.jpg">
      <figcaption id="right_caption_lf_basket" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 21.19
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
        <select id="left_select_lf_basket" onchange="select_model(this, 'left', 'lf', 'basket')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_lf_basket" type="range" min="0" max="84" value="0" onchange="select_frame(this, 'lf', 'basket')">
      </td>
      <td align="center">
        <select id="right_select_lf_basket" onchange="select_model(this, 'right', 'lf', 'basket')">
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

## Ship

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_lf_ship" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_lf_ship_220901/render_model/image000.jpg">
      <figcaption id="left_caption_lf_ship" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 26.26
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_lf_ship" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_lf_ship_220901/render_model/image000.jpg">
      <figcaption id="right_caption_lf_ship" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 30.16
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
        <select id="left_select_lf_ship" onchange="select_model(this, 'left', 'lf', 'ship')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_lf_ship" type="range" min="0" max="108" value="0" onchange="select_frame(this, 'lf', 'ship')">
      </td>
      <td align="center">
        <select id="right_select_lf_ship" onchange="select_model(this, 'right', 'lf', 'ship')">
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

## Statue

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_lf_statue" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_lf_statue_220901/render_model/image000.jpg">
      <figcaption id="left_caption_lf_statue" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 29.76
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_lf_statue" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_lf_statue_220901/render_model/image000.jpg">
      <figcaption id="right_caption_lf_statue" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 34.9
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
        <select id="left_select_lf_statue" onchange="select_model(this, 'left', 'lf', 'statue')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_lf_statue" type="range" min="0" max="75" value="0" onchange="select_frame(this, 'lf', 'statue')">
      </td>
      <td align="center">
        <select id="right_select_lf_statue" onchange="select_model(this, 'right', 'lf', 'statue')">
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

## Torch

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_lf_torch" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_lf_torch_220901/render_model/image000.jpg">
      <figcaption id="left_caption_lf_torch" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 23.24
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_lf_torch" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_lf_torch_220901/render_model/image000.jpg">
      <figcaption id="right_caption_lf_torch" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 25.86
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
        <select id="left_select_lf_torch" onchange="select_model(this, 'left', 'lf', 'torch')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_lf_torch" type="range" min="0" max="60" value="0" onchange="select_frame(this, 'lf', 'torch')">
      </td>
      <td align="center">
        <select id="right_select_lf_torch" onchange="select_model(this, 'right', 'lf', 'torch')">
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


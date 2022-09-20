---
layout: default
title: LLFF
parent: Visualization
nav_order: 3
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

## fern

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_fern" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_fern_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_fern" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 25.19
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_fern" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_fern_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_fern" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 24.58
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
        <select id="left_select_llff_fern" onchange="select_model(this, 'left', 'llff', 'fern')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_fern" type="range" min="0" max="2" value="0" onchange="select_frame(this, 'llff', 'fern')">
      </td>
      <td align="center">
        <select id="right_select_llff_fern" onchange="select_model(this, 'right', 'llff', 'fern')">
          <option value="nerf">NeRF</option>
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

## flower

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_flower" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_flower_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_flower" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 27.94
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_flower" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_flower_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_flower" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 27.81
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
        <select id="left_select_llff_flower" onchange="select_model(this, 'left', 'llff', 'flower')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_flower" type="range" min="0" max="4" value="0" onchange="select_frame(this, 'llff', 'flower')">
      </td>
      <td align="center">
        <select id="right_select_llff_flower" onchange="select_model(this, 'right', 'llff', 'flower')">
          <option value="nerf">NeRF</option>
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

## fortress

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_fortress" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_fortress_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_fortress" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 31.73
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_fortress" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_fortress_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_fortress" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 31.17
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
        <select id="left_select_llff_fortress" onchange="select_model(this, 'left', 'llff', 'fortress')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_fortress" type="range" min="0" max="5" value="0" onchange="select_frame(this, 'llff', 'fortress')">
      </td>
      <td align="center">
        <select id="right_select_llff_fortress" onchange="select_model(this, 'right', 'llff', 'fortress')">
          <option value="nerf">NeRF</option>
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

## horns

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_horns" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_horns_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_horns" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 28.03
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_horns" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_horns_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_horns" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 28.03
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
        <select id="left_select_llff_horns" onchange="select_model(this, 'left', 'llff', 'horns')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_horns" type="range" min="0" max="7" value="0" onchange="select_frame(this, 'llff', 'horns')">
      </td>
      <td align="center">
        <select id="right_select_llff_horns" onchange="select_model(this, 'right', 'llff', 'horns')">
          <option value="nerf">NeRF</option>
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

## leaves

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_leaves" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_leaves_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_leaves" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 21.17
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_leaves" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_leaves_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_leaves" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 20.28
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
        <select id="left_select_llff_leaves" onchange="select_model(this, 'left', 'llff', 'leaves')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_leaves" type="range" min="0" max="3" value="0" onchange="select_frame(this, 'llff', 'leaves')">
      </td>
      <td align="center">
        <select id="right_select_llff_leaves" onchange="select_model(this, 'right', 'llff', 'leaves')">
          <option value="nerf">NeRF</option>
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

## orchids

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_orchids" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_orchids_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_orchids" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 20.29
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_orchids" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_orchids_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_orchids" style="text-align:right;">
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
        <select id="left_select_llff_orchids" onchange="select_model(this, 'left', 'llff', 'orchids')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_orchids" type="range" min="0" max="3" value="0" onchange="select_frame(this, 'llff', 'orchids')">
      </td>
      <td align="center">
        <select id="right_select_llff_orchids" onchange="select_model(this, 'right', 'llff', 'orchids')">
          <option value="nerf">NeRF</option>
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

## room

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_room" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_room_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_room" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 32.96
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_room" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_room_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_room" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 33.55
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
        <select id="left_select_llff_room" onchange="select_model(this, 'left', 'llff', 'room')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_room" type="range" min="0" max="5" value="0" onchange="select_frame(this, 'llff', 'room')">
      </td>
      <td align="center">
        <select id="right_select_llff_room" onchange="select_model(this, 'right', 'llff', 'room')">
          <option value="nerf">NeRF</option>
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

## trex

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_llff_trex" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_llff_trex_220901/render_model/image000.jpg">
      <figcaption id="left_caption_llff_trex" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 27.52
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_llff_trex" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_llff_trex_220901/render_model/image000.jpg">
      <figcaption id="right_caption_llff_trex" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 27.87
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
        <select id="left_select_llff_trex" onchange="select_model(this, 'left', 'llff', 'trex')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_llff_trex" type="range" min="0" max="6" value="0" onchange="select_frame(this, 'llff', 'trex')">
      </td>
      <td align="center">
        <select id="right_select_llff_trex" onchange="select_model(this, 'right', 'llff', 'trex')">
          <option value="nerf">NeRF</option>
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

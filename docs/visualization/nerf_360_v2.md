---
layout: default
title: NeRF-360-v2
parent: Visualization
nav_order: 7
---

<script defer src="https://unpkg.com/img-comparison-slider@7/dist/index.js"></script>
<link rel="stylesheet" href="https://unpkg.com/img-comparison-slider@7/dist/styles.css"/>
<script src="../../../assets/js/my-js.js"></script>
<link rel="stylesheet" href="../../../assets/css/my-css.css"/>

---

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---
## bicycle

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_nerf_360_v2_bicycle" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_nerf_360_v2_bicycle_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="left_caption_nerf_360_v2_bicycle" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 21.82
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_nerf_360_v2_bicycle" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_nerf_360_v2_bicycle_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="right_caption_nerf_360_v2_bicycle" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 22.86
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
        <select id="left_select_nerf_360_v2_bicycle" onchange="select_model(this, 'left', 'nerf_360_v2', 'bicycle')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_nerf_360_v2_bicycle" type="range" min="0" max="19" value="0" onchange="select_frame(this, 'nerf_360_v2', 'bicycle')">
      </td>
      <td align="center">
        <select id="right_select_nerf_360_v2_bicycle" onchange="select_model(this, 'right', 'nerf_360_v2', 'bicycle')">
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

## bonsai

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_nerf_360_v2_bonsai" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_nerf_360_v2_bonsai_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="left_caption_nerf_360_v2_bonsai" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 29.03
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_nerf_360_v2_bonsai" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_nerf_360_v2_bonsai_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="right_caption_nerf_360_v2_bonsai" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 32.97
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
        <select id="left_select_nerf_360_v2_bonsai" onchange="select_model(this, 'left', 'nerf_360_v2', 'bonsai')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_nerf_360_v2_bonsai" type="range" min="0" max="29" value="0" onchange="select_frame(this, 'nerf_360_v2', 'bonsai')">
      </td>
      <td align="center">
        <select id="right_select_nerf_360_v2_bonsai" onchange="select_model(this, 'right', 'nerf_360_v2', 'bonsai')">
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

## counter

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_nerf_360_v2_counter" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_nerf_360_v2_counter_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="left_caption_nerf_360_v2_counter" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 26.98
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_nerf_360_v2_counter" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_nerf_360_v2_counter_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="right_caption_nerf_360_v2_counter" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 29.29
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
        <select id="left_select_nerf_360_v2_counter" onchange="select_model(this, 'left', 'nerf_360_v2', 'counter')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_nerf_360_v2_counter" type="range" min="0" max="23" value="0" onchange="select_frame(this, 'nerf_360_v2', 'counter')">
      </td>
      <td align="center">
        <select id="right_select_nerf_360_v2_counter" onchange="select_model(this, 'right', 'nerf_360_v2', 'counter')">
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

## garden

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_nerf_360_v2_garden" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_nerf_360_v2_garden_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="left_caption_nerf_360_v2_garden" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 23.64
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_nerf_360_v2_garden" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_nerf_360_v2_garden_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="right_caption_nerf_360_v2_garden" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 26.01
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
        <select id="left_select_nerf_360_v2_garden" onchange="select_model(this, 'left', 'nerf_360_v2', 'garden')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_nerf_360_v2_garden" type="range" min="0" max="18" value="0" onchange="select_frame(this, 'nerf_360_v2', 'garden')">
      </td>
      <td align="center">
        <select id="right_select_nerf_360_v2_garden" onchange="select_model(this, 'right', 'nerf_360_v2', 'garden')">
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

## kitchen

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_nerf_360_v2_kitchen" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_nerf_360_v2_kitchen_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="left_caption_nerf_360_v2_kitchen" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 27.16
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_nerf_360_v2_kitchen" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_nerf_360_v2_kitchen_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="right_caption_nerf_360_v2_kitchen" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 31.99
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
        <select id="left_select_nerf_360_v2_kitchen" onchange="select_model(this, 'left', 'nerf_360_v2', 'kitchen')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_nerf_360_v2_kitchen" type="range" min="0" max="27" value="0" onchange="select_frame(this, 'nerf_360_v2', 'kitchen')">
      </td>
      <td align="center">
        <select id="right_select_nerf_360_v2_kitchen" onchange="select_model(this, 'right', 'nerf_360_v2', 'kitchen')">
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

## room

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_nerf_360_v2_room" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_nerf_360_v2_room_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="left_caption_nerf_360_v2_room" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 30.1
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_nerf_360_v2_room" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_nerf_360_v2_room_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="right_caption_nerf_360_v2_room" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 32.68
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
        <select id="left_select_nerf_360_v2_room" onchange="select_model(this, 'left', 'nerf_360_v2', 'room')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_nerf_360_v2_room" type="range" min="0" max="31" value="0" onchange="select_frame(this, 'nerf_360_v2', 'room')">
      </td>
      <td align="center">
        <select id="right_select_nerf_360_v2_room" onchange="select_model(this, 'right', 'nerf_360_v2', 'room')">
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

## stump

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_nerf_360_v2_stump" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_nerf_360_v2_stump_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="left_caption_nerf_360_v2_stump" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 22.93
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_nerf_360_v2_stump" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf360_nerf_360_v2_stump_220901/render_model/image000.jpg" style="min-width: 900px;">
      <figcaption id="right_caption_nerf_360_v2_stump" style="text-align:right;">
        <b>Mip-NeRF 360</b><br>PSNR: 25.28
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
        <select id="left_select_nerf_360_v2_stump" onchange="select_model(this, 'left', 'nerf_360_v2', 'stump')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="nerfpp">NeRF++</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="mipnerf360">Mip-NeRF 360</option>
        </select>
      </td>
      <td align="center">
        <input id="input_nerf_360_v2_stump" type="range" min="0" max="12" value="0" onchange="select_frame(this, 'nerf_360_v2', 'stump')">
      </td>
      <td align="center">
        <select id="right_select_nerf_360_v2_stump" onchange="select_model(this, 'right', 'nerf_360_v2', 'stump')">
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


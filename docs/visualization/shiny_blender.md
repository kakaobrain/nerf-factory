---
layout: default
title: Shiny Blender
parent: Visualization
nav_order: 6
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

## ball

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_shiny_blender_ball" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_shiny_blender_ball_220901/render_model/image000.jpg">
      <figcaption id="left_caption_shiny_blender_ball" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 27.18
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_shiny_blender_ball" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_shiny_blender_ball_220901/render_model/image000.jpg">
      <figcaption id="right_caption_shiny_blender_ball" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 43.09
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
        <select id="left_select_shiny_blender_ball" onchange="select_model(this, 'left', 'shiny_blender', 'ball')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_shiny_blender_ball" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'shiny_blender', 'ball')">
      </td>
      <td align="center">
        <select id="right_select_shiny_blender_ball" onchange="select_model(this, 'right', 'shiny_blender', 'ball')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## car

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_shiny_blender_car" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_shiny_blender_car_220901/render_model/image000.jpg">
      <figcaption id="left_caption_shiny_blender_car" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 26.42
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_shiny_blender_car" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_shiny_blender_car_220901/render_model/image000.jpg">
      <figcaption id="right_caption_shiny_blender_car" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 30.7
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
        <select id="left_select_shiny_blender_car" onchange="select_model(this, 'left', 'shiny_blender', 'car')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_shiny_blender_car" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'shiny_blender', 'car')">
      </td>
      <td align="center">
        <select id="right_select_shiny_blender_car" onchange="select_model(this, 'right', 'shiny_blender', 'car')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## coffee

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_shiny_blender_coffee" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_shiny_blender_coffee_220901/render_model/image000.jpg">
      <figcaption id="left_caption_shiny_blender_coffee" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 30.64
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_shiny_blender_coffee" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_shiny_blender_coffee_220901/render_model/image000.jpg">
      <figcaption id="right_caption_shiny_blender_coffee" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 32.27
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
        <select id="left_select_shiny_blender_coffee" onchange="select_model(this, 'left', 'shiny_blender', 'coffee')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_shiny_blender_coffee" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'shiny_blender', 'coffee')">
      </td>
      <td align="center">
        <select id="right_select_shiny_blender_coffee" onchange="select_model(this, 'right', 'shiny_blender', 'coffee')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## helmet

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_shiny_blender_helmet" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_shiny_blender_helmet_220901/render_model/image000.jpg">
      <figcaption id="left_caption_shiny_blender_helmet" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 27.61
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_shiny_blender_helmet" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_shiny_blender_helmet_220901/render_model/image000.jpg">
      <figcaption id="right_caption_shiny_blender_helmet" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 29.66
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
        <select id="left_select_shiny_blender_helmet" onchange="select_model(this, 'left', 'shiny_blender', 'helmet')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_shiny_blender_helmet" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'shiny_blender', 'helmet')">
      </td>
      <td align="center">
        <select id="right_select_shiny_blender_helmet" onchange="select_model(this, 'right', 'shiny_blender', 'helmet')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## teapot

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_shiny_blender_teapot" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_shiny_blender_teapot_220901/render_model/image000.jpg">
      <figcaption id="left_caption_shiny_blender_teapot" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 45.37
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_shiny_blender_teapot" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_shiny_blender_teapot_220901/render_model/image000.jpg">
      <figcaption id="right_caption_shiny_blender_teapot" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 45.2
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
        <select id="left_select_shiny_blender_teapot" onchange="select_model(this, 'left', 'shiny_blender', 'teapot')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_shiny_blender_teapot" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'shiny_blender', 'teapot')">
      </td>
      <td align="center">
        <select id="right_select_shiny_blender_teapot" onchange="select_model(this, 'right', 'shiny_blender', 'teapot')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## toaster

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_shiny_blender_toaster" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_shiny_blender_toaster_220901/render_model/image000.jpg">
      <figcaption id="left_caption_shiny_blender_toaster" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 22.51
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_shiny_blender_toaster" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_shiny_blender_toaster_220901/render_model/image000.jpg">
      <figcaption id="right_caption_shiny_blender_toaster" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 24.88
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
        <select id="left_select_shiny_blender_toaster" onchange="select_model(this, 'left', 'shiny_blender', 'toaster')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_shiny_blender_toaster" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'shiny_blender', 'toaster')">
      </td>
      <td align="center">
        <select id="right_select_shiny_blender_toaster" onchange="select_model(this, 'right', 'shiny_blender', 'toaster')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---



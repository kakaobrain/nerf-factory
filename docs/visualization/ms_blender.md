---
layout: default
title: Multi-scale Blender
parent: Visualization
nav_order: 2
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

## chair

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_chair" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_chair_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="left_caption_blender_multiscale_chair" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 32.83
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_chair" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_chair_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="right_caption_blender_multiscale_chair" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 37.36
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
        <select id="left_select_blender_multiscale_chair" onchange="select_model(this, 'left', 'blender_multiscale', 'chair')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_chair" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'chair')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_chair" onchange="select_model(this, 'right', 'blender_multiscale', 'chair')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## drums

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_drums" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_drums_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="left_caption_blender_multiscale_drums" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 25.24
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_drums" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_drums_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="right_caption_blender_multiscale_drums" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 27.12
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
        <select id="left_select_blender_multiscale_drums" onchange="select_model(this, 'left', 'blender_multiscale', 'drums')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_drums" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'drums')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_drums" onchange="select_model(this, 'right', 'blender_multiscale', 'drums')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## ficus

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_ficus" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_ficus_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="left_caption_blender_multiscale_ficus" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 30.23
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_ficus" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_ficus_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="right_caption_blender_multiscale_ficus" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 33.0
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
        <select id="left_select_blender_multiscale_ficus" onchange="select_model(this, 'left', 'blender_multiscale', 'ficus')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_ficus" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'ficus')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_ficus" onchange="select_model(this, 'right', 'blender_multiscale', 'ficus')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## hotdog

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_hotdog" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_hotdog_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="left_caption_blender_multiscale_hotdog" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 35.24
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_hotdog" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_hotdog_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="right_caption_blender_multiscale_hotdog" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 39.36
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
        <select id="left_select_blender_multiscale_hotdog" onchange="select_model(this, 'left', 'blender_multiscale', 'hotdog')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_hotdog" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'hotdog')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_hotdog" onchange="select_model(this, 'right', 'blender_multiscale', 'hotdog')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## lego

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_lego" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_lego_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="left_caption_blender_multiscale_lego" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 31.45
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_lego" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_lego_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="right_caption_blender_multiscale_lego" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 35.71
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
        <select id="left_select_blender_multiscale_lego" onchange="select_model(this, 'left', 'blender_multiscale', 'lego')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_lego" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'lego')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_lego" onchange="select_model(this, 'right', 'blender_multiscale', 'lego')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## materials

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_materials" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_materials_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="left_caption_blender_multiscale_materials" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 29.54
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_materials" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_materials_220901/render_model/image000.jpg" style="min-width: 800px;">
      <figcaption id="right_caption_blender_multiscale_materials" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 32.63
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
        <select id="left_select_blender_multiscale_materials" onchange="select_model(this, 'left', 'blender_multiscale', 'materials')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_materials" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'materials')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_materials" onchange="select_model(this, 'right', 'blender_multiscale', 'materials')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## mic

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_mic" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_mic_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_multiscale_mic" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 32.2
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_mic" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_mic_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_multiscale_mic" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 37.93
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
        <select id="left_select_blender_multiscale_mic" onchange="select_model(this, 'left', 'blender_multiscale', 'mic')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_mic" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'mic')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_mic" onchange="select_model(this, 'right', 'blender_multiscale', 'mic')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

## ship

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_multiscale_ship" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_multiscale_ship_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_multiscale_ship" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 29.41
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_multiscale_ship" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/mipnerf_blender_multiscale_ship_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_multiscale_ship" style="text-align:right;">
        <b>Mip-NeRF</b><br>PSNR: 33.24
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
        <select id="left_select_blender_multiscale_ship" onchange="select_model(this, 'left', 'blender_multiscale', 'ship')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_multiscale_ship" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender_multiscale', 'ship')">
      </td>
      <td align="center">
        <select id="right_select_blender_multiscale_ship" onchange="select_model(this, 'right', 'blender_multiscale', 'ship')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option selected="selected" value="mipnerf">Mip-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---

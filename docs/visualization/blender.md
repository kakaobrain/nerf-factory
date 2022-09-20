---
layout: default
title: Blender
parent: Visualization
nav_order: 1
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
      <img id="left_img_blender_chair" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_chair_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_chair" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 34.93
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_chair" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_chair_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_chair" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 35.84
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
        <select id="left_select_blender_chair" onchange="select_model(this, 'left', 'blender', 'chair')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_chair" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'chair')">
      </td>
      <td align="center">
        <select id="right_select_blender_chair" onchange="select_model(this, 'right', 'blender', 'chair')">
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

## drums

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_drums" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_drums_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_drums" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 25.28
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_drums" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_drums_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_drums" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 25.52
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
        <select id="left_select_blender_drums" onchange="select_model(this, 'left', 'blender', 'drums')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_drums" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'drums')">
      </td>
      <td align="center">
        <select id="right_select_blender_drums" onchange="select_model(this, 'right', 'blender', 'drums')">
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

## ficus

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_ficus" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_ficus_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_ficus" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 31.28
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_ficus" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_ficus_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_ficus" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 31.32
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
        <select id="left_select_blender_ficus" onchange="select_model(this, 'left', 'blender', 'ficus')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_ficus" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'ficus')">
      </td>
      <td align="center">
        <select id="right_select_blender_ficus" onchange="select_model(this, 'right', 'blender', 'ficus')">
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

## hotdog

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_hotdog" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_hotdog_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_hotdog" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 37.16
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_hotdog" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_hotdog_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_hotdog" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 36.54
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
        <select id="left_select_blender_hotdog" onchange="select_model(this, 'left', 'blender', 'hotdog')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_hotdog" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'hotdog')">
      </td>
      <td align="center">
        <select id="right_select_blender_hotdog" onchange="select_model(this, 'right', 'blender', 'hotdog')">
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

## lego

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_lego" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_lego_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_lego" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 34.38
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_lego" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_lego_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_lego" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 35.79
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
        <select id="left_select_blender_lego" onchange="select_model(this, 'left', 'blender', 'lego')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxels</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_lego" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'lego')">
      </td>
      <td align="center">
        <select id="right_select_blender_lego" onchange="select_model(this, 'right', 'blender', 'lego')">
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

## materials

<div align="center">
  <img-comparison-slider hover="hover">
    <figure slot="first" class="before">
      <img id="left_img_blender_materials" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_materials_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_materials" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 30.45
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_materials" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_materials_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_materials" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 35.71
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
        <select id="left_select_blender_materials" onchange="select_model(this, 'left', 'blender', 'materials')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxel</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_materials" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'materials')">
      </td>
      <td align="center">
        <select id="right_select_blender_materials" onchange="select_model(this, 'right', 'blender', 'materials')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxel</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
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
      <img id="left_img_blender_mic" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_mic_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_mic" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 35.18
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_mic" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_mic_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_mic" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 35.96
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
        <select id="left_select_blender_mic" onchange="select_model(this, 'left', 'blender', 'mic')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxel</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_mic" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'mic')">
      </td>
      <td align="center">
        <select id="right_select_blender_mic" onchange="select_model(this, 'right', 'blender', 'mic')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxel</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
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
      <img id="left_img_blender_ship" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/nerf_blender_ship_220901/render_model/image000.jpg">
      <figcaption id="left_caption_blender_ship" style="text-align:left;">
        <b>NeRF</b><br>PSNR: 29.95
      </figcaption>
    </figure>
    <figure slot="second" class="after">
      <img id="right_img_blender_ship" src="https://huggingface.co/nrtf/nerf_factory/resolve/main/refnerf_blender_ship_220901/render_model/image000.jpg">
      <figcaption id="right_caption_blender_ship" style="text-align:right;">
        <b>Ref-NeRF</b><br>PSNR: 29.51
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
        <select id="left_select_blender_ship" onchange="select_model(this, 'left', 'blender', 'ship')">
          <option selected="selected" value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxel</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option value="refnerf">Ref-NeRF</option>
        </select>
      </td>
      <td align="center">
        <input id="input_blender_ship" type="range" min="0" max="199" value="0" onchange="select_frame(this, 'blender', 'ship')">
      </td>
      <td align="center">
        <select id="right_select_blender_ship" onchange="select_model(this, 'right', 'blender', 'ship')">
          <option value="nerf">NeRF</option>
          <option value="plenoxel">Plenoxel</option>
          <option value="dvgo">DVGO</option>
          <option value="mipnerf">Mip-NeRF</option>
          <option selected="selected" value="refnerf">Ref-NeRF</option>
        </select>
      </td>
    </tr>
  </table>
</div>

---


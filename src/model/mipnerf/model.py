# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Mip-NeRF (https://github.com/google/mipnerf)
# Copyright (c) 2021 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import src.model.mipnerf.helper as helper
import utils.store_image as store_image
from src.model.interface import LitModel


@gin.configurable()
class MipNeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(MipNeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view
        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    """
    x: torch.Tensor, [batch, num_samples, feature]
    condition: torch.Tensor, [batch, feature]
    """

    def forward(self, x, condition):
        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

        return raw_rgb, raw_density


@gin.configurable()
class MipNeRF(nn.Module):
    def __init__(
        self,
        num_samples: int = 128,
        num_levels: int = 2,
        resample_padding: float = 0.01,
        stop_level_grad: bool = True,
        use_viewdirs: bool = True,
        lindisp: bool = False,
        ray_shape: str = "cone",
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        deg_view: int = 4,
        density_noise: float = 0,
        density_bias: float = -1,
        rgb_padding: float = 0.001,
        disable_integration: bool = False,
    ):
        # Layers
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(MipNeRF, self).__init__()
        self.density_activation = nn.Softplus()
        self.rgb_activation = nn.Sigmoid()

        self.mlp = MipNeRFMLP(min_deg_point, max_deg_point, deg_view)

    def forward(self, rays, randomized, white_bkgd, near, far):

        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    radii=rays["radii"],
                    num_samples=self.num_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    ray_shape=self.ray_shape,
                )
            else:
                t_vals, samples = helper.resample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    radii=rays["radii"],
                    t_vals=t_vals,
                    weights=weights,
                    randomized=randomized,
                    ray_shape=self.ray_shape,
                    stop_level_grad=self.stop_level_grad,
                    resample_padding=self.resample_padding,
                )

            samples_enc = helper.integrated_pos_enc(
                samples=samples, min_deg=self.min_deg_point, max_deg=self.max_deg_point
            )

            viewdirs_enc = helper.pos_enc(
                rays["viewdirs"], min_deg=0, max_deg=self.deg_view, append_identity=True
            )

            raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)

            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.rand_like(raw_density)

            rgb = self.rgb_activation(raw_rgb)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights = helper.volumetric_rendering(
                rgb, density, t_vals, rays["rays_d"], white_bkgd=white_bkgd
            )

            ret.append((comp_rgb, distance, acc))

        return ret


@gin.configurable()
class LitMipNeRF(LitModel):
    def __init__(
        self,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        coarse_loss_mult: float = 0.1,
        randomized: bool = True,
        use_multiscale: bool = False,
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitMipNeRF, self).__init__()
        self.model = MipNeRF()

    def setup(self, stage):
        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.white_bkgd = self.trainer.datamodule.white_bkgd

    def training_step(self, batch, batch_idx):
        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]
        target = batch["target"]

        mask = batch["multloss"] if self.use_multiscale else None

        loss0 = helper.img2mse(rgb_coarse, target, mask)
        loss1 = helper.img2mse(rgb_fine, target, mask)
        loss = loss1 + loss0 * self.coarse_loss_mult

        with torch.no_grad():
            if self.use_multiscale:
                loss0 = helper.img2mse(rgb_coarse, target, None)
                loss1 = helper.img2mse(rgb_fine, target, None)

            psnr0 = helper.mse2psnr(loss0)
            psnr1 = helper.mse2psnr(loss1)

        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)

        return loss

    def render_rays(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far
        )
        rgb_fine = rendered_results[1][0]
        target = batch["target"]
        ret["target"] = target
        ret["rgb"] = rgb_fine
        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = gin.query_parameter("run.max_steps")

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def validation_epoch_end(self, outputs):
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)

            result_path = os.path.join(self.logdir, "results.json")
            self.write_stats(result_path, psnr, ssim, lpips)

        return psnr, ssim, lpips

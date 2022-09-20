# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Ref-NeRF (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from typing import Any, Callable

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import src.model.refnerf.helper as helper
import src.model.refnerf.ref_utils as ref_utils
import utils.store_image as store_image
from src.model.interface import LitModel


@gin.configurable()
class RefNeRFMLP(nn.Module):
    def __init__(
        self,
        deg_view,
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        netdepth: int = 8,
        netwidth: int = 256,
        bottleneck_width: int = 128,
        netdepth_viewdirs: int = 8,
        netwidth_viewdirs: int = 256,
        # net_activation: Callable[..., Any] = nn.ReLU(),
        skip_layer: int = 4,
        skip_layer_dir: int = 4,
        perturb: float = 1.0,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        num_roughness_channels: int = 1,
        # roughness_activation: Callable[..., Any] = nn.Softplus(),
        roughness_bias: float = -1.0,
        bottleneck_noise: float = 0.0,
        # density_activation: Callable[..., Any] = nn.Softplus(),
        density_bias: float = -1.0,
        density_noise: float = 0.0,
        rgb_premultiplier: float = 1.0,
        # rgb_activation: Callable[..., Any] = nn.Sigmoid(),
        rgb_bias: float = 0.0,
        rgb_padding: float = 0.001,
        num_normal_channels: int = 3,
        num_tint_channels: int = 3,
        # tint_activation: Callable[..., Any] = nn.Sigmoid(),
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(RefNeRFMLP, self).__init__()

        self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)

        self.net_activation = nn.ReLU()
        self.roughness_activation = nn.Softplus()
        self.density_activation = nn.Softplus()
        self.rgb_activation = nn.Sigmoid()
        self.tint_activation = nn.Sigmoid()

        pos_size = ((max_deg_point - min_deg_point) * 2) * input_ch
        view_pos_size = (2**deg_view - 1 + deg_view) * 2
        init_layer = nn.Linear(pos_size, self.netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(self.netdepth - 1):
            if idx % self.skip_layer == 0 and idx > 0:
                module = nn.Linear(self.netwidth + pos_size, self.netwidth)
            else:
                module = nn.Linear(self.netwidth, self.netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [
            nn.Linear(self.bottleneck_width + view_pos_size + 1, self.netwidth_viewdirs)
        ]
        for idx in range(self.netdepth_viewdirs - 1):
            if idx % self.skip_layer_dir == 0 and idx > 0:
                module = nn.Linear(
                    self.netwidth_viewdirs + self.bottleneck_width + view_pos_size + 1,
                    self.netwidth_viewdirs,
                )
            else:
                module = nn.Linear(self.netwidth_viewdirs, self.netwidth_viewdirs)
            init.xavier_uniform_(module.weight)
            views_linear.append(module)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(self.netwidth, self.bottleneck_width)
        self.density_layer = nn.Linear(self.netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(self.netwidth_viewdirs, num_rgb_channels)

        self.normal_layer = nn.Linear(self.netwidth, self.num_normal_channels)
        self.rgb_diffuse_layer = nn.Linear(self.netwidth, self.num_rgb_channels)
        self.tint_layer = nn.Linear(self.netwidth, self.num_tint_channels)
        self.roughness_layer = nn.Linear(self.netwidth, self.num_roughness_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.normal_layer.weight)
        init.xavier_uniform_(self.rgb_diffuse_layer.weight)
        init.xavier_uniform_(self.tint_layer.weight)
        init.xavier_uniform_(self.roughness_layer.weight)

    """
    x: torch.Tensor, [batch, num_samples, feature]
    viewdirs: torch.Tensor, [batch, viewdirs]
    """

    def forward(self, samples, viewdirs):

        means, covs = samples

        with torch.set_grad_enabled(True):
            means.requires_grad_(True)
            x = helper.integrated_pos_enc(
                means=means,
                covs=covs,
                min_deg=self.min_deg_point,
                max_deg=self.max_deg_point,
            )
            num_samples, feat_dim = x.shape[1:]
            x = x.reshape(-1, feat_dim)

            inputs = x
            for idx in range(self.netdepth):
                x = self.pts_linears[idx](x)
                x = self.net_activation(x)
                if idx % self.skip_layer == 0 and idx > 0:
                    x = torch.cat([x, inputs], dim=-1)

            raw_density = self.density_layer(x)

            raw_density_grad = torch.autograd.grad(
                outputs=raw_density.sum(), inputs=means, retain_graph=True
            )[0]

            raw_density_grad = raw_density_grad.reshape(
                -1, num_samples, self.num_normal_channels
            )

            normals = -ref_utils.l2_normalize(raw_density_grad)
            means.detach()

        density = self.density_activation(raw_density + self.density_bias)
        density = density.reshape(-1, num_samples, self.num_density_channels)

        grad_pred = self.normal_layer(x).reshape(
            -1, num_samples, self.num_normal_channels
        )
        normals_pred = -ref_utils.l2_normalize(grad_pred)
        normals_to_use = normals_pred

        raw_rgb_diffuse = self.rgb_diffuse_layer(x)

        tint = self.tint_layer(x)
        tint = self.tint_activation(tint)

        raw_roughness = self.roughness_layer(x)
        roughness = self.roughness_activation(raw_roughness + self.roughness_bias)
        roughness = roughness.reshape(-1, num_samples, self.num_roughness_channels)

        bottleneck = self.bottleneck_layer(x)
        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)
        bottleneck = bottleneck.reshape(-1, num_samples, self.bottleneck_width)

        refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
        dir_enc = self.dir_enc_fn(refdirs, roughness)

        dotprod = torch.sum(
            normals_to_use * viewdirs[..., None, :], dim=-1, keepdims=True
        )

        x = torch.cat([bottleneck, dir_enc, dotprod], dim=-1)
        x = x.reshape(-1, x.shape[-1])
        inputs = x
        for idx in range(self.netdepth_viewdirs):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer_dir == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_rgb = self.rgb_layer(x)
        rgb = self.rgb_activation(self.rgb_premultiplier * raw_rgb + self.rgb_bias)

        diffuse_linear = self.rgb_activation(raw_rgb_diffuse - np.log(3.0))
        specular_linear = tint * rgb
        rgb = torch.clamp(
            helper.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0
        )

        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        rgb = rgb.reshape(-1, num_samples, self.num_rgb_channels)

        return dict(
            rgb=rgb,
            density=density,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )


@gin.configurable()
class RefNeRF(nn.Module):
    def __init__(
        self,
        num_samples: int = 128,
        num_levels: int = 2,
        resample_padding: float = 0.01,
        stop_level_grad: bool = True,
        use_viewdirs: bool = True,
        lindisp: bool = False,
        ray_shape: str = "cone",
        deg_view: int = 5,
        rgb_padding: float = 0.001,
    ):
        # Layers
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(RefNeRF, self).__init__()

        self.mlp = RefNeRFMLP(self.deg_view)

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

            ray_results = self.mlp(samples, rays["viewdirs"])
            comp_rgb, distance, acc, weights = helper.volumetric_rendering(
                ray_results["rgb"],
                ray_results["density"],
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            rendered_result = ray_results
            rendered_result["comp_rgb"] = comp_rgb
            rendered_result["distance"] = distance
            rendered_result["acc"] = acc
            rendered_result["weights"] = weights

            ret.append(rendered_result)

        return ret


@gin.configurable()
class LitRefNeRF(LitModel):
    def __init__(
        self,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        coarse_loss_mult: float = 0.1,
        randomized: bool = True,
        orientation_loss_mult: float = 0.1,
        orientation_coarse_loss_mult: float = 0.01,
        predicted_normal_loss_mult: float = 3e-4,
        predicted_normal_coarse_loss_mult: float = 3e-5,
        compute_normal_metrics: bool = False,
        grad_max_norm: float = 0.001,
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitRefNeRF, self).__init__()
        self.model = RefNeRF()

    def setup(self, stage):
        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.white_bkgd = self.trainer.datamodule.white_bkgd

    def training_step(self, batch, batch_idx):
        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )

        rgb_coarse = rendered_results[0]["comp_rgb"]
        rgb_fine = rendered_results[1]["comp_rgb"]
        target = batch["target"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)
        loss = loss1 + loss0 * self.coarse_loss_mult

        if self.compute_normal_metrics:
            normal_mae = self.compute_normal_mae(rendered_results, batch["normals"])
            self.log("train/normal_mae", normal_mae, on_step=True)

        if self.orientation_coarse_loss_mult > 0 or self.orientation_loss_mult > 0:
            orientation_loss = self.orientation_loss(
                rendered_results, batch["viewdirs"]
            )
            self.log("train/orientation_loss", orientation_loss, on_step=True)
            loss += orientation_loss

        if (
            self.predicted_normal_coarse_loss_mult > 0
            or self.predicted_normal_loss_mult > 0
        ):
            pred_normal_loss = self.predicted_normal_loss(rendered_results)
            self.log("train/pred_normal_loss", pred_normal_loss, on_step=True)
            loss += pred_normal_loss

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
        rgb_fine = rendered_results[1]["comp_rgb"]
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

        if self.grad_max_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_max_norm)

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

    def orientation_loss(self, rendered_results, viewdirs):
        total_loss = 0.0
        for i, rendered_result in enumerate(rendered_results):
            w = rendered_result["weights"]
            n = rendered_result["normals_pred"]
            if n is None:
                raise ValueError("Normals cannot be None if orientation loss is on.")
            v = -1.0 * viewdirs
            n_dot_v = (n * v[..., None, :]).sum(axis=-1)
            loss = torch.mean(
                (w * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)
            )
            if i < self.model.num_levels - 1:
                total_loss += self.orientation_coarse_loss_mult * loss
            else:
                total_loss += self.orientation_loss_mult * loss
        return total_loss

    def predicted_normal_loss(self, rendered_results):
        total_loss = 0.0
        for i, rendered_result in enumerate(rendered_results):
            w = rendered_result["weights"]
            n = rendered_result["normals"]
            n_pred = rendered_result["normals_pred"]
            if n is None or n_pred is None:
                raise ValueError(
                    "Predicted normals and gradient normals cannot be None if "
                    "predicted normal loss is on."
                )
            loss = torch.mean((w * (1.0 - torch.sum(n * n_pred, dim=-1))).sum(dim=-1))
            if i < self.model.num_levels - 1:
                total_loss += self.predicted_normal_coarse_loss_mult * loss
            else:
                total_loss += self.predicted_normal_loss_mult * loss
        return total_loss

    def compute_normal_mae(self, rendered_results, normals):
        normals_gt, alphas = torch.split(normals, [3, 1], dim=-1)
        weights = rendered_results[1]["weights"] * alphas
        normalized_normals_gt = ref_utils.l2_normalize(normals_gt[..., None, :])
        normalized_normals = ref_utils.l2_normalize(rendered_results[1]["normals"])
        normal_mae = ref_utils.compute_weighted_mae(
            weights, normalized_normals, normalized_normals_gt
        )
        return normal_mae

# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Plenoxels (https://github.com/sxyu/svox2)
# Copyright (c) 2022 the Plenoxel authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from typing import *

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import src.model.plenoxel.dataclass as dataclass
import src.model.plenoxel.sparse_grid as sparse_grid
import src.model.plenoxel.utils as utils
import utils.store_image as store_image
from src.model.interface import LitModel
from src.model.plenoxel.__global__ import BASIS_TYPE_SH


class PlenoxelOptim(torch.optim.Optimizer):
    def step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        return loss


@gin.configurable()
class ResampleCallBack(pl.Callback):
    def __init__(self, upsamp_every):
        self.upsamp_every = upsamp_every

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            pl_module.curr_step > 0
            and pl_module.curr_step % self.upsamp_every == 0
            and pl_module.reso_idx + 1 < len(pl_module.reso_list)
        ):
            if pl_module.tv_early_only:
                pl_module.lambda_tv = 0.0
                pl_module.lambda_tv_sh = 0.0
            elif pl_module.tv_decay != 1.0:
                pl_module.lambda_tv *= pl_module.tv_decay
                pl_module.lambda_tv_sh *= pl_module.tv_decay

            intrinsics = trainer.datamodule.intrinsics
            extrinsics = trainer.datamodule.extrinsics
            image_sizes = trainer.datamodule.image_sizes

            # NDC should be updated.
            camera_list = (
                pl_module.generate_camera_list(
                    intrinsics, extrinsics, None, image_sizes
                )
                if pl_module.thresh_type == "weight"
                else None
            )

            pl_module.reso_idx += 1
            reso = pl_module.reso_list[pl_module.reso_idx]
            pl_module.model.resample(
                reso=reso,
                sigma_thresh=pl_module.density_thresh,
                weight_thresh=pl_module.weight_thresh / reso[2],
                dilate=2,
                cameras=camera_list,
                max_elements=pl_module.max_grid_elements,
            )

            if pl_module.model.use_background and pl_module.reso_idx <= 1:
                pl_module.model.sparsify_background(pl_module.background_density_thresh)

            if pl_module.upsample_density_add:
                pl_module.model.density_data.data[:] += pl_module.upsample_density_add


@gin.configurable()
class LitPlenoxel(LitModel):

    # The external dataset will be called.
    def __init__(
        self,
        reso: List[List[int]] = [[256, 256, 256], [512, 512, 512]],
        upsample_step: List[int] = [38400, 76800],
        init_iters: int = 0,
        upsample_density_add: float = 0.0,
        basis_type: str = "sh",
        sh_dim: int = 9,
        mlp_posenc_size: int = 4,
        mlp_width: int = 32,
        background_nlayers: int = 0,
        background_reso: int = 512,
        # Sigma Optim
        sigma_optim: str = "rmsprop",
        lr_sigma: float = 3e1,
        lr_sigma_final: float = 5e-2,
        lr_sigma_decay_steps: int = 250000,
        lr_sigma_delay_steps: int = 15000,
        lr_sigma_delay_mult: float = 1e-2,
        # SH Optim
        sh_optim: str = "rmsprop",
        lr_sh: float = 1e-2,
        lr_sh_final: float = 5e-6,
        lr_sh_decay_steps: int = 250000,
        lr_sh_delay_steps: int = 0,
        lr_sh_delay_mult: float = 1e-2,
        lr_fg_begin_step: int = 0,
        # BG Simga Optim
        bg_optim: str = "rmsprop",
        lr_sigma_bg: float = 3e0,
        lr_sigma_bg_final: float = 3e-3,
        lr_sigma_bg_decay_steps: int = 250000,
        lr_sigma_bg_delay_steps: int = 0,
        lr_sigma_bg_delay_mult: float = 1e-2,
        # BG Colors Optim
        lr_color_bg: float = 1e-1,
        lr_color_bg_final: float = 5e-6,
        lr_color_bg_decay_steps: int = 250000,
        lr_color_bg_delay_steps: int = 0,
        lr_color_bg_delay_mult: float = 1e-2,
        # Basis Optim
        basis_optim: str = "rmsprop",
        lr_basis: float = 1e-6,
        lr_basis_final: float = 1e-6,
        lr_basis_decay_steps: int = 250000,
        lr_basis_delay_steps: int = 0,
        lr_basis_begin_step: int = 0,
        lr_basis_delay_mult: float = 1e-2,
        # RMSProp Option
        rms_beta: float = 0.95,
        # Init Option
        init_sigma: float = 0.1,
        init_sigma_bg: float = 0.1,
        thresh_type: str = "weight",
        weight_thresh: float = 0.0005 * 512,
        density_thresh: float = 5.0,
        background_density_thresh: float = 1.0 + 1e-9,
        max_grid_elements: int = 44_000_000,
        tune_mode: bool = False,
        tune_nosave: bool = False,
        # Losses
        lambda_tv: float = 1e-5,
        tv_sparsity: float = 0.01,
        tv_logalpha: bool = False,
        lambda_tv_sh: float = 1e-3,
        tv_sh_sparsity: float = 0.01,
        lambda_tv_lumisphere: float = 0.0,
        tv_lumisphere_sparsity: float = 0.01,
        tv_lumisphere_dir_factor: float = 0.0,
        tv_decay: float = 1.0,
        lambda_l2_sh: float = 0.0,
        tv_early_only: int = 1,
        tv_contiguous: int = 1,
        # Other Lambdas
        lambda_sparsity: float = 0.0,
        lambda_beta: float = 0.0,
        lambda_tv_background_sigma: float = 1e-2,
        lambda_tv_background_color: float = 1e-2,
        tv_background_sparsity: float = 0.01,
        # WD
        weight_decay_sigma: float = 1.0,
        weight_decay_sh: float = 1.0,
        lr_decay: bool = True,
        n_train: Optional[int] = None,
        nosphereinit: bool = False,
        # Render Options
        step_size: float = 0.5,
        sigma_thresh: float = 1e-8,
        stop_thresh: float = 1e-7,
        background_brightness: float = 1.0,
        renderer_backend: str = "cuvol",
        random_sigma_std: float = 0.0,
        random_sigma_std_background: float = 0.0,
        near_clip: float = 0.00,
        use_spheric_clip: bool = False,
        enable_random: bool = False,
        last_sample_opaque: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitPlenoxel, self).__init__()

        assert basis_type in ["sh", "3d_texture", "mlp"]
        assert sigma_optim in ["sgd", "rmsprop"]
        assert sh_optim in ["sgd", "rmsprop"]
        assert bg_optim in ["sgd", "rmsprop"]
        assert basis_optim in ["sgd", "rmsprop"]
        assert thresh_type in ["weight", "sigma"]
        assert renderer_backend in ["cuvol"]

        self.automatic_optimization = False
        self.reso_idx = 0
        self.reso_list = reso
        self.lr_sigma_func = self.get_expon_lr_func(
            lr_sigma,
            lr_sigma_final,
            lr_sigma_delay_steps,
            lr_sigma_delay_mult,
            lr_sigma_decay_steps,
        )
        self.lr_sh_func = self.get_expon_lr_func(
            lr_sh,
            lr_sh_final,
            lr_sh_delay_steps,
            lr_sh_delay_mult,
            lr_sh_decay_steps,
        )
        self.lr_sigma_bg_func = self.get_expon_lr_func(
            lr_sigma_bg,
            lr_sigma_bg_final,
            lr_sigma_bg_delay_steps,
            lr_sigma_bg_delay_mult,
            lr_sigma_bg_decay_steps,
        )
        self.lr_color_bg_func = self.get_expon_lr_func(
            lr_color_bg,
            lr_color_bg_final,
            lr_color_bg_delay_steps,
            lr_color_bg_delay_mult,
            lr_color_bg_decay_steps,
        )
        self.curr_step = 0

    def setup(self, stage: Optional[str] = None) -> None:

        dmodule = self.trainer.datamodule

        self.model = sparse_grid.SparseGrid(
            reso=self.reso_list[self.reso_idx],
            center=dmodule.scene_center,
            radius=dmodule.scene_radius,
            use_sphere_bound=dmodule.use_sphere_bound and not self.nosphereinit,
            basis_dim=self.sh_dim,
            use_z_order=True,
            basis_type=eval("BASIS_TYPE_" + self.basis_type.upper()),
            mlp_posenc_size=self.mlp_posenc_size,
            mlp_width=self.mlp_width,
            background_nlayers=self.background_nlayers,
            background_reso=self.background_reso,
            device=self.device,
        )

        if stage is None or stage == "fit":
            self.model.sh_data.data[:] = 0.0
            self.model.density_data.data[:] = (
                0.0 if self.lr_fg_begin_step > 0 else self.init_sigma
            )
            if self.model.use_background:
                self.model.background_data.data[..., -1] = self.init_sigma_bg

        self.ndc_coeffs = dmodule.ndc_coeffs
        return super().setup(stage)

    def generate_camera_list(
        self, intrinsics=None, extrinsics=None, ndc_coeffs=None, image_size=None
    ):
        dmodule = self.trainer.datamodule
        return [
            dataclass.Camera(
                torch.from_numpy(
                    self.extrinsics[i] if extrinsics is None else extrinsics[i]
                ).to(dtype=torch.float32, device=self.device),
                self.intrinsics[0, 0] if intrinsics is None else intrinsics[i, 0, 0],
                self.intrinsics[1, 1] if intrinsics is None else intrinsics[i, 1, 1],
                self.intrinsics[0, 2] if intrinsics is None else intrinsics[i, 0, 2],
                self.intrinsics[1, 2] if intrinsics is None else intrinsics[i, 1, 2],
                self.w if image_size is None else image_size[i, 0],
                self.h if image_size is None else image_size[i, 1],
                self.ndc_coeffs if ndc_coeffs is None else ndc_coeffs[i],
            )
            for i in dmodule.i_train
        ]

    def get_expon_lr_func(
        self, lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps
    ):
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        return helper

    def configure_optimizers(self):
        return PlenoxelOptim(self.parameters(), {})

    def training_step(self, batch, batch_idx):
        gstep = self.curr_step

        def closure():

            if self.lr_fg_begin_step > 0 and gstep == self.lr_fg_begin_step:
                self.model.density_data.data[:] = self.init_sigma

            # Plenoxel only supports Float32
            rays_o = batch["rays_o"].to(torch.float32)
            rays_d = batch["rays_d"].to(torch.float32)
            target = batch["target"].to(torch.float32)

            rays = dataclass.Rays(rays_o.contiguous(), rays_d.contiguous())

            rgb = self.model.volume_render_fused(
                rays,
                target,
                beta_loss=self.lambda_beta,
                sparsity_loss=self.lambda_sparsity,
                randomize=self.enable_random,
            )

            img_loss = utils.img2mse(rgb, target)
            psnr = utils.mse2psnr(img_loss)

            self.log("train_psnr", psnr, on_step=True, prog_bar=True, logger=True)

            if self.lambda_tv > 0.0:
                self.model.inplace_tv_grad(
                    self.model.density_data.grad,
                    scaling=self.lambda_tv,
                    sparse_frac=self.tv_sparsity,
                    logalpha=self.tv_logalpha,
                    ndc_coeffs=self.ndc_coeffs,
                    contiguous=self.tv_contiguous,
                )

            if self.lambda_tv_sh > 0.0:
                self.model.inplace_tv_color_grad(
                    self.model.sh_data.grad,
                    scaling=self.lambda_tv_sh,
                    sparse_frac=self.tv_sh_sparsity,
                    ndc_coeffs=self.ndc_coeffs,
                    contiguous=self.tv_contiguous,
                )

            if self.lambda_tv_lumisphere > 0.0:
                self.model.inplace_tv_lumisphere_grad(
                    self.model.sh_data.grad,
                    scaling=self.lambda_tv_lumisphere,
                    dir_factor=self.tv_lumisphere_dir_factor,
                    sparse_frac=self.tv_lumisphere_sparsity,
                    ndc_coeffs=self.ndc_coeffs,
                )

            if self.lambda_l2_sh > 0.0:
                self.model.inplace_l2_color_grad(
                    self.model.sh_data.grad, scaling=self.lambda_l2_sh
                )

            if self.model.use_background and (
                self.lambda_tv_background_sigma > 0.0
                or self.lambda_tv_background_color > 0.0
            ):
                self.model.inplace_tv_background_grad(
                    self.model.background_data.grad,
                    scaling=self.lambda_tv_background_color,
                    scaling_density=self.lambda_tv_background_sigma,
                    sparse_frac=self.tv_background_sparsity,
                    contiguous=self.tv_contiguous,
                )

            return img_loss

        # Dummy update for step counting
        self.optimizers().step(closure)

        gstep = self.global_step
        lr_sigma = self.lr_sigma_func(gstep)
        lr_sh = self.lr_sh_func(gstep)
        lr_sigma_bg = self.lr_sigma_bg_func(gstep - self.lr_basis_begin_step)
        lr_color_bg = self.lr_color_bg_func(gstep - self.lr_basis_begin_step)

        if gstep >= self.lr_fg_begin_step:
            self.model.optim_density_step(
                lr_sigma, beta=self.rms_beta, optim=self.sigma_optim
            )
            self.model.optim_sh_step(lr_sh, beta=self.rms_beta, optim=self.sh_optim)

        if self.model.use_background:
            self.model.optim_background_step(
                lr_sigma_bg, lr_color_bg, beta=self.rms_beta, optim=self.bg_optim
            )

        if self.weight_decay_sh < 1.0 and gstep % 20 == 0:
            self.model.sh_data.data *= self.weight_decay_sigma
        if self.weight_decay_sigma < 1.0 and gstep % 20 == 0:
            self.model.density_data.data *= self.weight_decay_sh

    def render_rays(
        self,
        batch,
        batch_idx,
        prefix="",
        cpu=False,
        randomize=False,
    ):
        ret = {}
        rays_o = batch["rays_o"].to(torch.float32)
        rays_d = batch["rays_d"].to(torch.float32)
        if "target" in batch.keys():
            target = batch["target"].to(torch.float32)
        else:
            target = (
                torch.zeros((len(rays_o), 3), dtype=torch.float32, device=self.device)
                + 0.5
            )

        rays = dataclass.Rays(rays_o.contiguous(), rays_d.contiguous())
        rgb = self.model.volume_render_fused(
            rays,
            target,
            beta_loss=self.lambda_beta,
            sparsity_loss=self.lambda_sparsity,
            randomize=randomize,
        )
        depth = self.model.volume_render_depth(
            rays,
            self.model.opt.sigma_thresh,
        )
        if cpu:
            rgb = rgb.detach().cpu()
            depth = depth.detach().cpu()
            target = target.detach().cpu()

        rgb_key, depth_key, target_key = "rgb", "depth", "target"
        if prefix != "":
            rgb_key = f"{prefix}/{rgb_key}"
            depth_key = f"{prefix}/{depth_key}"
            target_key = f"{prefix}/{target_key}"

        ret[rgb_key] = rgb
        ret[depth_key] = depth[:, None]
        if "target" in batch.keys():
            ret[target_key] = target

        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

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

            self.write_stats(
                os.path.join(self.logdir, "results.json"), psnr, ssim, lpips
            )

    def validation_epoch_end(self, outputs):
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True)
        return super().validation_epoch_end(outputs)

    def on_save_checkpoint(self, checkpoint) -> None:

        checkpoint["reso_idx"] = self.reso_idx
        density_data = checkpoint["state_dict"]["model.density_data"]

        sh = checkpoint["state_dict"]["model.sh_data"]

        model_links = checkpoint["state_dict"]["model.links"].cpu()
        reso_list = self.reso_list[self.reso_idx]

        argsort_val = model_links[torch.where(model_links >= 0)].argsort()
        links_compressed = torch.stack(torch.where(model_links >= 0))[:, argsort_val]
        links_idx = (
            reso_list[1] * reso_list[2] * links_compressed[0]
            + reso_list[2] * links_compressed[1]
            + links_compressed[2]
        )

        links = -torch.ones_like(model_links, device="cpu")
        links[
            links_compressed[0], links_compressed[1], links_compressed[2]
        ] = torch.arange(len(links_compressed[0]), dtype=torch.int32, device="cpu")

        background_data = checkpoint["state_dict"]["model.background_data"].cpu()

        if self.model.use_background:
            checkpoint["state_dict"].pop("model.background_data")

        checkpoint["state_dict"]["model.background_data"] = background_data

        checkpoint["state_dict"].pop("model.density_data")
        checkpoint["state_dict"].pop("model.sh_data")
        checkpoint["state_dict"].pop("model.links")

        checkpoint["state_dict"]["model.density_data"] = density_data
        checkpoint["state_dict"]["model.links_idx"] = links_idx
        checkpoint["state_dict"]["model.sh_data"] = sh

        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:

        state_dict = checkpoint["state_dict"]

        self.reso_idx = checkpoint["reso_idx"]

        del self.model.basis_data
        del self.model.density_data
        del self.model.sh_data
        del self.model.links

        self.model.register_parameter(
            "basis_data", nn.Parameter(state_dict["model.basis_data"])
        )

        if "model.background_data" in state_dict.keys():
            del self.model.background_data
            bgd_data = state_dict["model.background_data"]

            self.model.register_parameter("background_data", nn.Parameter(bgd_data))
            checkpoint["state_dict"]["model.background_data"] = bgd_data

        density_data = state_dict["model.density_data"]

        self.model.register_parameter("density_data", nn.Parameter(density_data))
        checkpoint["state_dict"]["model.density_data"] = density_data

        sh_data = state_dict["model.sh_data"]

        self.model.register_parameter("sh_data", nn.Parameter(sh_data))
        checkpoint["state_dict"]["model.sh_data"] = sh_data

        reso = self.reso_list[checkpoint["reso_idx"]]

        links = torch.zeros(reso, dtype=torch.int32) - 1
        links_sparse = state_dict["model.links_idx"]
        links_idx = torch.stack(
            [
                links_sparse // (reso[1] * reso[2]),
                links_sparse % (reso[1] * reso[2]) // reso[2],
                links_sparse % reso[2],
            ]
        ).long()
        links[links_idx[0], links_idx[1], links_idx[2]] = torch.arange(
            len(links_idx[0]), dtype=torch.int32
        )
        checkpoint["state_dict"].pop("model.links_idx")
        checkpoint["state_dict"]["model.links"] = links
        self.model.register_buffer("links", links)

        return super().on_load_checkpoint(checkpoint)

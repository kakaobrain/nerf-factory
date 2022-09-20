# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from DVGO (https://github.com/sunset1995/DirectVoxGO)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import time
from typing import *

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tqdm
from torch_efficient_distloss import flatten_eff_distloss

import src.model.dvgo.dcvgo as dcvgo
import src.model.dvgo.dmpigo as dmpigo
import src.model.dvgo.dvgo as dvgo
import src.model.dvgo.utils as utils
import utils.store_image as store_image
from src.model.dvgo.__global__ import *
from src.model.interface import LitModel


class Coarse2Fine(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        global_step = pl_module.global_step
        if global_step == pl_module.N_iters_coarse and pl_module.model_step == "coarse":

            print("Starts fine training!")

            coarse_ckpt_path = f"{pl_module.logdir}/coarse_last.ckpt"
            torch.save(
                {
                    "global_step": global_step,
                    "model_kwargs": pl_module.model.get_kwargs(),
                    "model_state_dict": pl_module.model.state_dict(),
                    "optimizer_state_dict": pl_module.optimizer.state_dict(),
                },
                coarse_ckpt_path,
            )

            pl_module.model_step = "fine"
            pl_module.update_config("fine")
            xyz_min_fine, xyz_max_fine = pl_module.compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO,
                model_path=coarse_ckpt_path,
                thres=pl_module._bbox_thres,
            )

            pl_module.xyz_min_fine = xyz_min_fine
            pl_module.xyz_max_fine = xyz_max_fine

            (
                pl_module.model,
                optimizer,
                pl_module.cfg_train,
                _,
                _,
                _,
                _,
                pl_module.render_kwargs,
            ) = pl_module.scene_rep_reconstruction(
                xyz_min=xyz_min_fine.cpu().numpy(),
                xyz_max=xyz_max_fine.cpu().numpy(),
                stage="fine",
                coarse_ckpt_path=coarse_ckpt_path,
            )
            trainer.optimizers = [optimizer]
            pl_module.optimizer = optimizer

            if pl_module.ray_masking:
                rays_o = pl_module.trainer.datamodule.train_dataloader().dataset.rays_o
                rays_d = pl_module.trainer.datamodule.train_dataloader().dataset.rays_d
                viewdirs = (
                    pl_module.trainer.datamodule.train_dataloader().dataset.viewdirs
                )

                device = pl_module.device
                bsz = pl_module.trainer.datamodule.batch_size
                num_elems = rays_o.shape[0]
                mask = torch.zeros(num_elems).to(device=device)
                for i in tqdm.tqdm(range(0, num_elems, bsz)):
                    mask[i : i + bsz] = pl_module.model.hit_coarse_geo(
                        rays_o=torch.from_numpy(rays_o)[i : i + bsz].to(device=device),
                        rays_d=torch.from_numpy(rays_d)[i : i + bsz].to(device=device),
                        viewdirs=torch.from_numpy(viewdirs)[i : i + bsz].to(
                            device=device
                        ),
                        **pl_module.get_render_kwargs(),
                    )

                mask = mask.detach().cpu().numpy().astype(np.bool)

                pl_module.trainer.datamodule.update_masked_sampler(mask)

            print("Start fine training")

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


class UpdateOccupancyMask(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        global_step = pl_module.global_step
        if pl_module.model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            pl_module.model.update_occupancy_cache()

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


class ProgressiveScaling(pl.Callback):
    def setup(self, trainer, pl_module, stage) -> None:

        if stage == "fit" or stage is None:
            self.pg_scale = (
                np.array(pl_module.pg_scale_fine) + pl_module.N_iters_coarse
            ).tolist()
        return super().setup(trainer, pl_module, stage)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):

        global_step = pl_module.global_step

        if global_step in self.pg_scale:
            n_rest_scales = len(self.pg_scale) - self.pg_scale.index(global_step) - 1
            cur_voxels = int(pl_module._num_voxels / (2**n_rest_scales))
            print(f"Currnet voxel: {cur_voxels}")
            if isinstance(
                pl_module.model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)
            ):
                pl_module.model.scale_volume_grid(cur_voxels)
            elif isinstance(pl_module.model, (dmpigo.DirectMPIGO)):
                pl_module.model.scale_volume_grid(cur_voxels, pl_module._mpi_depth)
            else:
                raise NotImplementedError
            cfg_train = pl_module.get_train_kwargs()
            optimizer = utils.create_optimizer_or_freeze_model(
                pl_module.model, cfg_train, global_step=0
            )
            trainer.optimizers = [optimizer]
            pl_module.optimizer = optimizer
            pl_module.model.act_shift -= pl_module._decay_after_scale
            torch.cuda.empty_cache()

        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)


@gin.configurable()
class LitDVGO(LitModel):
    def __init__(
        self,
        model_type: str = "dvgo",
        ray_masking: bool = False,
        bbox_type="bounded",
        ###### coarse training
        N_iters_coarse: int = 5000,  # number of optimization steps
        N_rand_coarse: int = 8192,
        lrate_density_coarse: float = 1e-1,  # lr of density voxel grid
        lrate_k0_coarse: float = 1e-1,  # lr of color/feature voxel grid
        lrate_rgbnet_coarse: float = 1e-3,  # lr of the mlp to preduct view-dependent color
        lrate_decay_coarse: int = 20,  # lr decay by 0.1 after every lrate_decay*1000 steps
        pervoxel_lr_coarse: bool = True,  # view-count-based lr
        pervoxel_lr_downrate_coarse: int = 1,  # downsampled image for computing view-count-based lr
        ray_sampler_coarse: str = "random",  # ray sampling strategies
        weight_main_coarse: float = 1.0,  # weight of photometric loss
        weight_entropy_last_coarse: float = 0.01,  # weight of background entropy loss
        weight_nearclip_coarse: int = 0,
        weight_distortion_coarse: int = 0,
        weight_rgbper_coarse: float = 0.1,  # weight of per-point rgb loss
        tv_every_coarse: int = 1,  # count total variation loss every tv_every step
        tv_after_coarse: int = 0,  # count total variation loss from tv_from step
        tv_before_coarse: int = 0,  # count total variation before the given number of iterations
        tv_dense_before_coarse: int = 0,  # count total variation densely before the given number of iterations
        weight_tv_density_coarse: float = 0.0,  # weight of total variation loss of density voxel grid
        weight_tv_k0_coarse: float = 0.0,  # weight of total variation loss of color/feature voxel grid
        pg_scale_coarse: List[int] = [],  # checkpoints for progressive scaling
        decay_after_scale_coarse: float = 1.0,  # decay act_shift after scaling
        skip_zero_grad_fields_coarse: List[
            str
        ] = [],  # the variable name to skip optimizing parameters w/ zero grad in each iteration
        maskout_lt_nviews_coarse: int = 0,
        ##_coarse model rendering related
        num_voxels_coarse: int = 1024000,  # expected number of voxel
        num_voxels_base_coarse: int = 1024000,  # to rescale delta distance
        density_type_coarse: str = "DenseGrid",  # DenseGrid, TensoRFGrid
        k0_type_coarse: str = "DenseGrid",  # DenseGrid, TensoRFGrid
        mpi_depth_coarse: int = 128,  # the number of planes in Multiplane Image (work when ndc=True)
        nearest_coarse: bool = False,  # nearest interpolation
        pre_act_density_coarse: bool = False,  # pre-activated trilinear interpolation
        in_act_density_coarse: bool = False,  # in-activated trilinear interpolation
        bbox_thres_coarse: float = 1e-3,  # threshold to determine known free-space in the fine stage
        mask_cache_thres_coarse: float = 1e-3,  # threshold to determine a tighten BBox in the fine stage
        rgbnet_dim_coarse: int = 0,  # feature voxel grid dim
        rgbnet_full_implicit_coarse: bool = False,  # let the colors MLP ignore feature voxel grid
        rgbnet_direct_coarse: bool = True,  # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
        rgbnet_depth_coarse: int = 3,  # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
        rgbnet_width_coarse: int = 128,  # width of the colors MLP
        alpha_init_coarse: float = 1e-6,  # set the alpha values everywhere at the begin of training
        fast_color_thres_coarse: float = 1e-7,  # threshold of alpha value to skip the fine stage sampled point
        maskout_near_cam_vox_coarse: bool = True,  # maskout grid points that between cameras and their near planes
        world_bound_scale_coarse: int = 1,  # rescale the BBox enclosing the scene
        stepsize_coarse: float = 0.5,  # sampling stepsize in volume rendering
        _mask_cache_world_size_coarse: Any = None,
        _density_config_coarse=dict(),
        _k0_config_coarse=dict(),
        i_weights_coarse: int = 750,
        mask_cache_world_size_coarse: Any = None,
        density_config_coarse=dict(),
        k0_config_coarse=dict(),
        viewbase_pe_coarse: int = 4,
        # fine training related
        N_rand_fine: int = 8192,
        lrate_density_fine: float = 1e-1,  # lr of density voxel grid
        lrate_k0_fine: float = 1e-1,  # lr of color/feature voxel grid
        lrate_rgbnet_fine: float = 1e-3,  # lr of the mlp to preduct view-dependent color
        lrate_decay_fine: int = 20,  # lr decay by 0.1 after every lrate_decay*1000 steps
        pervoxel_lr_fine: bool = False,  # view-count-based lr
        pervoxel_lr_downrate_fine: int = 1,  # downsampled image for computing view-count-based lr
        ray_sampler_fine: str = "in_maskcache",  # ray sampling strategies
        weight_main_fine: float = 1.0,  # weight of photometric loss
        weight_entropy_last_fine: float = 0.001,  # weight of background entropy loss
        weight_nearclip_fine: int = 0,
        weight_distortion_fine: int = 0,
        weight_rgbper_fine: float = 0.01,  # weight of per-point rgb loss
        tv_every_fine: int = 1,  # count total variation loss every tv_every step
        tv_after_fine: int = 0,  # count total variation loss from tv_from step
        tv_before_fine: int = 0,  # count total variation before the given number of iterations
        tv_dense_before_fine: int = 0,  # count total variation densely before the given number of iterations
        weight_tv_density_fine: float = 0.0,  # weight of total variation loss of density voxel grid
        weight_tv_k0_fine: float = 0.0,  # weight of total variation loss of color/feature voxel grid
        pg_scale_fine: List[int] = [
            1000,
            2000,
            3000,
            4000,
        ],  # checkpoints for progressive scaling
        decay_after_scale_fine: float = 1.0,  # decay act_shift after scaling
        skip_zero_grad_fields_fine: List[str] = [
            "density",
            "k0",
        ],  # the variable name to skip optimizing parameters w/ zero grad in each iteration
        maskout_lt_nviews_fine: int = 0,
        # fine model related
        num_voxels_fine: int = 160**3,  # expected number of voxel
        num_voxels_base_fine: int = 160**3,  # to rescale delta distance
        density_type_fine: str = "DenseGrid",  # DenseGrid, TensoRFGrid
        k0_type_fine: str = "DenseGrid",  # DenseGrid, TensoRFGrid
        mpi_depth_fine: int = 128,  # the number of planes in Multiplane Image (work when ndc=True)
        nearest_fine: bool = False,  # nearest interpolation
        pre_act_density_fine: bool = False,  # pre-activated trilinear interpolation
        in_act_density_fine: bool = False,  # in-activated trilinear interpolation
        bbox_thres_fine: float = 1e-3,  # threshold to determine known free-space in the fine stage
        mask_cache_thres_fine: float = 1e-3,  # threshold to determine a tighten BBox in the fine stage
        rgbnet_dim_fine: int = 12,  # feature voxel grid dim
        rgbnet_full_implicit_fine: bool = False,  # let the colors MLP ignore feature voxel grid
        rgbnet_direct_fine: bool = True,  # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
        rgbnet_depth_fine: int = 3,  # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
        rgbnet_width_fine: int = 128,  # width of the colors MLP
        alpha_init_fine: float = 1e-2,  # set the alpha values everywhere at the begin of training
        fast_color_thres_fine: float = 1e-4,  # threshold of alpha value to skip the fine stage sampled point
        maskout_near_cam_vox_fine: bool = False,  # maskout grid points that between cameras and their near planes
        world_bound_scale_fine: int = 1.05,  # rescale the BBox enclosing the scene
        stepsize_fine: float = 0.5,  # sampling stepsize in volume rendering
        i_weights_fine: int = 750,
        mask_cache_world_size_fine: Any = None,
        density_config_fine=dict(),
        k0_config_fine=dict(),
        viewbase_pe_fine: int = 4,
        _load2gpu_on_the_fly: bool = False,  # do not load all images into gpu (to save gpu memory)
        rand_bkgd: bool = False,  # use random background during training
        contracted_norm_fine: str = "l2",
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        init()

        if fast_color_thres_fine == "outdoor_default":
            self.fast_color_thres_fine = {
                "_delete_": True,
                0: self.alpha_init_fine * self.stepsize_fine / 10,
                1500: min(self.alpha_init_fine, 1e-4) * self.stepsize_fine / 5,
                2500: min(self.alpha_init_fine, 1e-4) * self.stepsize_fine / 2,
                3500: min(self.alpha_init_fine, 1e-4) * self.stepsize_fine / 1.5,
                4500: min(self.alpha_init_fine, 1e-4) * self.stepsize_fine,
                5500: min(self.alpha_init_fine, 1e-4),
                6500: 1e-4,
            }

        super(LitDVGO, self).__init__()

    def compute_bbox_by_cam_frustrm_bounded(self):
        rays_o = self.trainer.datamodule.train_dataloader().dataset.rays_o
        rays_d = self.trainer.datamodule.train_dataloader().dataset.rays_d

        if self.ndc_coord:
            pts_nf = np.stack(
                [rays_o + rays_d * self.near, rays_o + rays_d * self.far], 1
            )
        else:
            viewdirs = self.trainer.datamodule.train_dataloader().dataset.viewdirs
            pts_nf = np.stack(
                [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far], 1
            )

        xyz_min = np.amin(pts_nf, axis=(0, 1))
        xyz_max = np.amax(pts_nf, axis=(0, 1))
        return xyz_min, xyz_max

    def compute_bbox_by_cam_frustrm_unbounded(self):
        rays_o = self.trainer.datamodule.train_dataloader().dataset.rays_o
        rays_d = self.trainer.datamodule.train_dataloader().dataset.rays_d
        near_clip = self.trainer.datamodule.near_clip

        pts = rays_o + rays_d * near_clip
        xyz_min, xyz_max = np.min(pts, axis=0), np.max(pts, axis=0)
        center = (xyz_min + xyz_max) * 0.5
        radius = (center - xyz_min).max()
        xyz_min = center - radius
        xyz_max = center + radius

        return xyz_min, xyz_max

    def setup(self, stage):

        self.N_iters_fine = self.trainer.max_steps - self.N_iters_coarse

        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.ndc_coord = self.trainer.datamodule.ndc_coord

        self.white_bkgd = self.trainer.datamodule.white_bkgd

        self.ndc_coeffs = self.trainer.datamodule.ndc_coeffs
        self.Ks = self.trainer.datamodule.intrinsics
        self.c2ws = self.trainer.datamodule.extrinsics
        self.HWs = self.trainer.datamodule.image_sizes
        self.i_train = self.trainer.datamodule.i_train

        if self.bbox_type == "bounded":
            bbox_fun = self.compute_bbox_by_cam_frustrm_bounded
        elif self.bbox_type == "unbounded_inward":
            bbox_fun = self.compute_bbox_by_cam_frustrm_unbounded

        self.xyz_min_coarse, self.xyz_max_coarse = bbox_fun()

        self.N_train_img = len(self.i_train)

        print("compute_bbox_by_cam_frustrm: xyz_min", self.xyz_min_coarse)
        print("compute_bbox_by_cam_frustrm: xyz_max", self.xyz_max_coarse)

        if stage == "fit" or stage is None:
            coarse_or_fine = "coarse" if self.N_iters_coarse != 0 else "fine"
            (
                self.model,
                self.optimizer,
                _,
                _,
                _,
                _,
                _,
                self.render_kwargs,
            ) = self.scene_rep_reconstruction(
                xyz_min=self.xyz_min_coarse,
                xyz_max=self.xyz_max_coarse,
                stage=coarse_or_fine,
            )
            self.model_step = coarse_or_fine

        elif stage == "test" or stage is None:
            (
                self.model,
                _,
                _,
                _,
                _,
                _,
                _,
                self.render_kwargs,
            ) = self.scene_rep_reconstruction(
                xyz_min=self.xyz_min_coarse,
                xyz_max=self.xyz_max_coarse,
                stage="fine",
                coarse_ckpt_path=os.path.join(self.logdir, "coarse_last.ckpt"),
            )
            self.model_step = "fine"

        self.render_kwargs = self.get_render_kwargs()

    def configure_optimizers(self):
        return self.optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):

        global_step = self.global_step

        rays_o = batch["rays_o"]
        rays_d = batch["rays_d"]
        viewdirs = batch["viewdirs"]
        target = batch["target"]

        # volume rendering
        render_result = self.model(
            rays_o,
            rays_d,
            viewdirs,
            global_step=global_step,
            is_train=True,
            **self.render_kwargs,
        )

        rgbloss = self._weight_main * F.mse_loss(
            render_result["rgb_marched"], target.float()
        )
        loss = rgbloss
        if self._weight_entropy_last > 0:
            pout = render_result["alphainv_last"].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(
                pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)
            ).mean()
            loss += self._weight_entropy_last * entropy_last_loss
        if self._weight_nearclip > 0:
            near_thres = (
                self.trainer.datamodule.near_clip / self.model.scene_radius[0].item()
            )
            near_mask = render_result["t"] < near_thres
            density = render_result["raw_density"][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += self._weight_nearclip * nearclip_loss
        if self._weight_distortion > 0:
            n_max = render_result["n_max"]
            s = render_result["s"]
            w = render_result["weights"]
            ray_id = render_result["ray_id"]
            loss_distortion = flatten_eff_distloss(
                w, s.to(w.device), 1 / n_max, ray_id.to(w.device)
            )
            loss += self._weight_distortion * loss_distortion
        if self._weight_rgbper > 0:
            rgbper = (
                (render_result["raw_rgb"] - target[render_result["ray_id"]])
                .pow(2)
                .sum(-1)
            )
            rgbper_loss = (rgbper * render_result["weights"].detach()).sum() / len(
                rays_o
            )
            loss += self._weight_rgbper * rgbper_loss
        psnr = utils.mse2psnr(rgbloss.detach())

        self.log("train/loss", loss.item(), prog_bar=True, on_step=True)
        self.log("train/psnr", psnr.item(), prog_bar=True, on_step=True)

        return loss

    def on_after_backward(self) -> None:
        global_step = self.global_step
        if (
            global_step < self._tv_before
            and global_step > self._tv_after
            and global_step % self._tv_every == 0
        ):
            if self._weight_tv_density > 0:
                self.model.density_total_variation_add_grad(
                    self._weight_tv_density / self.trainer.datamodule.batch_size,
                    global_step < self._tv_dense_before,
                )
            if self._weight_tv_k0 > 0:
                self.model.k0_total_variation_add_grad(
                    self._weight_tv_k0 / self.trainer.datamodule.batch_size,
                    global_step < self._tv_dense_before,
                )
        return super().on_after_backward()

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ) -> None:

        optimizer.step(optimizer_closure)

        decay_steps = self._lrate_decay * 1000
        decay_factor = 0.1 ** (1 / decay_steps)

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * decay_factor

    def render_rays(self, batch, batch_idx):

        ret = {}
        rays_o = batch["rays_o"]
        rays_d = batch["rays_d"]
        viewdirs = batch["viewdirs"]
        target = batch["target"]

        render_result = self.model(
            rays_o, rays_d, viewdirs, is_train=False, **self.get_render_kwargs()
        )

        ret["rgb"] = render_result["rgb_marched"]
        if "target" in batch:
            ret["target"] = target
        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

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

    @torch.no_grad()
    def compute_bbox_by_coarse_geo(self, model_class, model_path, thres):
        print("compute_bbox_by_coarse_geo: start")
        eps_time = time.time()
        model = utils.load_model(model_class, model_path)
        interp = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, model.world_size[0]),
                torch.linspace(0, 1, model.world_size[1]),
                torch.linspace(0, 1, model.world_size[2]),
            ),
            -1,
        )
        dense_xyz = model.xyz_min * (1 - interp) + model.xyz_max * interp
        density = model.density(dense_xyz)
        alpha = model.activate_density(density)
        mask = alpha > thres
        active_xyz = dense_xyz[mask]
        xyz_min = active_xyz.amin(0)
        xyz_max = active_xyz.amax(0)
        eps_time = time.time() - eps_time
        print("compute_bbox_by_coarse_geo: xyz_min", xyz_min)
        print("compute_bbox_by_coarse_geo: xyz_max", xyz_max)
        print("compute_bbox_by_coarse_geo: finish (eps time:", eps_time, "secs)")
        return xyz_min, xyz_max

    def scene_rep_reconstruction(self, xyz_min, xyz_max, stage, coarse_ckpt_path=None):

        # init
        self.update_config(stage)
        cfg_train = self.get_train_kwargs()
        near = self.near
        far = self.far
        poses = self.c2ws
        i_train = self.i_train

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if abs(self._world_bound_scale - 1) > 1e-9:
            xyz_shift = (xyz_max - xyz_min) * (self._world_bound_scale - 1) / 2
            xyz_min -= xyz_shift
            xyz_max += xyz_shift

        # find whether there is existing checkpoint path
        last_ckpt_name = "last.tar" if stage == "fine" else "coarse_last.tar"
        last_ckpt_path = os.path.join(self.logdir, last_ckpt_name)
        reload_ckpt_path = None

        # init model and optimizer
        if reload_ckpt_path is None:
            print(f"scene_rep_reconstruction ({stage}): train from scratch")

            local_num_voxels = self._num_voxels
            if len(self._pg_scale):
                local_num_voxels = int(local_num_voxels / (2 ** len(self._pg_scale)))

            print(
                f"scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m"
            )

            model_kwargs = dict(
                xyz_min=xyz_min,
                xyz_max=xyz_max,
                num_voxels=local_num_voxels,
                num_voxels_base=self._num_voxels_base,
                alpha_init=self._alpha_init,
                mask_cache_thres=self._mask_cache_thres,
                mask_cache_world_size=self._mask_cache_world_size,
                fast_color_thres=self._fast_color_thres,
                density_type=self._density_type,
                k0_type=self._k0_type,
                density_config=self._density_config,
                k0_config=self._k0_config,
                rgbnet_dim=self._rgbnet_dim,
                rgbnet_direct=self._rgbnet_direct,
                rgbnet_full_implicit=self._rgbnet_full_implicit,
                rgbnet_depth=self._rgbnet_depth,
                rgbnet_width=self._rgbnet_width,
                viewbase_pe=self._viewbase_pe,
                mpi_depth=self._mpi_depth,
            )

            if self.model_type == "dvgo":
                self.model_fun = dvgo.DirectVoxGO
                model_kwargs["mask_cache_path"] = coarse_ckpt_path
            elif self.model_type == "dmpigo":
                self.model_fun = dmpigo.DirectMPIGO
            elif self.model_type == "dcvgo":
                model_kwargs["contracted_norm"] = self.contracted_norm_fine
                self.model_fun = dcvgo.DirectContractedVoxGO

            model = self.model_fun(**model_kwargs)
            model = model.to(device)
            optimizer = utils.create_optimizer_or_freeze_model(
                model, cfg_train, global_step=0
            )

            if self._maskout_near_cam_vox:
                model.maskout_near_cam_vox(poses[i_train, :3, 3], near)
        else:
            print(f"scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}")

        rays_o = self.trainer.datamodule.train_dataloader().dataset.rays_o
        rays_d = self.trainer.datamodule.train_dataloader().dataset.rays_d
        hws = self.HWs[self.i_train]

        ## modified from original scene_rep
        render_kwargs = self.get_render_kwargs()

        rays_o_tr, rays_d_tr, cnt = [], [], 0
        for (H, W) in hws:
            rays_o_tr.append(rays_o[cnt : cnt + H * W].reshape(H, W, 3))
            rays_d_tr.append(rays_d[cnt : cnt + H * W].reshape(H, W, 3))
            cnt += H * W
        imsz = [1] * len(self.HWs[self.i_train])

        # view-count-based learning rate
        if self._pervoxel_lr:

            def per_voxel_init():
                cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr,
                    rays_d_tr=rays_d_tr,
                    imsz=imsz,
                    near=self.near,
                    far=self.far,
                    stepsize=self._stepsize,
                    downrate=self._pervoxel_lr_downrate,
                    irregular_shape=False,
                )
                optimizer.set_pervoxel_lr(cnt)

                model.mask_cache.mask[cnt.squeeze() <= 2] = False

            per_voxel_init()

        if self._maskout_lt_nviews > 0:
            model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, cfg_train, self._maskout_lt_nviews
            )

        return (
            model,
            optimizer,
            cfg_train,
            None,
            rays_o_tr,
            rays_d_tr,
            last_ckpt_path,
            render_kwargs,
        )

    def update_config(self, stage):

        # training related
        self._N_iters = eval("self.N_iters_" + stage)
        self._N_rand = eval("self.N_rand_" + stage)
        self._lrate_density = eval("self.lrate_density_" + stage)
        self._lrate_k0 = eval("self.lrate_k0_" + stage)
        self._lrate_rgbnet = eval("self.lrate_rgbnet_" + stage)
        self._lrate_decay = eval("self.lrate_decay_" + stage)
        self._pervoxel_lr = eval("self.pervoxel_lr_" + stage)
        self._pervoxel_lr_downrate = eval("self.pervoxel_lr_downrate_" + stage)
        self._ray_sampler = eval("self.ray_sampler_" + stage)
        self._weight_main = eval("self.weight_main_" + stage)
        self._weight_entropy_last = eval("self.weight_entropy_last_" + stage)
        self._weight_nearclip = eval("self.weight_nearclip_" + stage)
        self._weight_distortion = eval("self.weight_distortion_" + stage)
        self._weight_rgbper = eval("self.weight_rgbper_" + stage)
        self._tv_every = eval("self.tv_every_" + stage)
        self._tv_after = eval("self.tv_after_" + stage)
        self._tv_before = eval("self.tv_before_" + stage)
        self._tv_dense_before = eval("self.tv_dense_before_" + stage)
        self._weight_tv_density = eval("self.weight_tv_density_" + stage)
        self._weight_tv_k0 = eval("self.weight_tv_k0_" + stage)
        self._pg_scale = eval("self.pg_scale_" + stage)
        self._decay_after_scale = eval("self.decay_after_scale_" + stage)
        self._skip_zero_grad_fields = eval("self.skip_zero_grad_fields_" + stage)
        self._maskout_lt_nviews = eval("self.maskout_lt_nviews_" + stage)
        # model related
        self._num_voxels = eval("self.num_voxels_" + stage)
        self._num_voxels_base = eval("self.num_voxels_base_" + stage)
        self._density_type = eval("self.density_type_" + stage)
        self._k0_type = eval("self.k0_type_" + stage)
        self._mpi_depth = eval("self.mpi_depth_" + stage)
        self._nearest = eval("self.nearest_" + stage)
        self._pre_act_density = eval("self.pre_act_density_" + stage)
        self._in_act_density = eval("self.in_act_density_" + stage)
        self._bbox_thres = eval("self.bbox_thres_" + stage)
        self._mask_cache_thres = eval("self.mask_cache_thres_" + stage)
        self._rgbnet_dim = eval("self.rgbnet_dim_" + stage)
        self._rgbnet_full_implicit = eval("self.rgbnet_full_implicit_" + stage)
        self._rgbnet_direct = eval("self.rgbnet_direct_" + stage)
        self._rgbnet_depth = eval("self.rgbnet_depth_" + stage)
        self._rgbnet_width = eval("self.rgbnet_width_" + stage)
        self._alpha_init = eval("self.alpha_init_" + stage)
        self._fast_color_thres = eval("self.fast_color_thres_" + stage)
        self._maskout_near_cam_vox = eval("self.maskout_near_cam_vox_" + stage)
        self._world_bound_scale = eval("self.world_bound_scale_" + stage)
        self._stepsize = eval("self.stepsize_" + stage)
        self._mask_cache_world_size = eval("self.mask_cache_world_size_" + stage)
        self._density_config = eval("self.density_config_" + stage)
        self._k0_config = eval("self.k0_config_" + stage)
        self._viewbase_pe = eval("self.viewbase_pe_" + stage)

    def get_train_kwargs(
        self,
    ):
        return {
            "N_iters": self._N_iters,
            "N_rand": self._N_rand,
            "lrate_density": self._lrate_density,
            "lrate_k0": self._lrate_k0,
            "lrate_rgbnet": self._lrate_rgbnet,
            "lrate_decay": self._lrate_decay,
            "pervoxel_lr": self._pervoxel_lr,
            "pervoxel_lr_downrate": self._pervoxel_lr_downrate,
            "ray_sampler": self._ray_sampler,
            "weight_main": self._weight_main,
            "weight_entropy_last": self._weight_entropy_last,
            "weight_nearclip": self._weight_nearclip,
            "weight_distortion": self._weight_distortion,
            "weight_rgbper": self._weight_rgbper,
            "tv_every": self._tv_every,
            "tv_after": self._tv_after,
            "tv_before": self._tv_before,
            "tv_dense_before": self._tv_dense_before,
            "weight_tv_density": self._weight_tv_density,
            "weight_tv_k0": self._weight_tv_k0,
            "pg_scale": self._pg_scale,
            "decay_after_scale": self._decay_after_scale,
            "skip_zero_grad_fields": self._skip_zero_grad_fields,
            "maskout_lt_nviews": self._maskout_lt_nviews,
        }

    def get_render_kwargs(
        self,
    ):
        # init rendering setup
        render_kwargs = {
            "near": self.near,
            "far": self.far,
            "bg": 1 if self.white_bkgd else 0,
            "rand_bkgd": self.rand_bkgd,
            "stepsize": self._stepsize,
        }
        return render_kwargs

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        np.save(os.path.join(self.logdir, "cfg_model"), self.model.get_kwargs())
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:

        del self.model

        model_kwargs = np.load(
            os.path.join(self.logdir, "cfg_model.npy"), allow_pickle=True
        ).item()

        if self.model_type == "dvgo":
            self.model = dvgo.DirectVoxGO(**model_kwargs)
        elif self.model_type == "dmpigo":
            self.model = dmpigo.DirectMPIGO(**model_kwargs)
        else:
            self.model = dcvgo.DirectContractedVoxGO(**model_kwargs)

        return super().on_load_checkpoint(checkpoint)

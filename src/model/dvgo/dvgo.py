# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from DVGO (https://github.com/sunset1995/DirectVoxGO)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_coo

from . import grid

"""Model"""


class DirectVoxGO(torch.nn.Module):
    def __init__(
        self,
        xyz_min,
        xyz_max,
        num_voxels=0,
        num_voxels_base=0,
        alpha_init=None,
        mask_cache_path=None,
        mask_cache_thres=1e-3,
        mask_cache_world_size=None,
        fast_color_thres=0,
        density_type="DenseGrid",
        k0_type="DenseGrid",
        density_config={},
        k0_config={},
        rgbnet_dim=0,
        rgbnet_direct=False,
        rgbnet_full_implicit=False,
        rgbnet_depth=3,
        rgbnet_width=128,
        viewbase_pe=4,
        **kwargs
    ):
        super(DirectVoxGO, self).__init__()
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = (
            (self.xyz_max - self.xyz_min).prod() / self.num_voxels_base
        ).pow(1 / 3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer(
            "act_shift", torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)])
        )
        print("dvgo: set density bias shift to", self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
            density_type,
            channels=1,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
            config=self.density_config,
        )

        # init color representation
        self.rgbnet_kwargs = {
            "rgbnet_dim": rgbnet_dim,
            "rgbnet_direct": rgbnet_direct,
            "rgbnet_full_implicit": rgbnet_full_implicit,
            "rgbnet_depth": rgbnet_depth,
            "rgbnet_width": rgbnet_width,
            "viewbase_pe": viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config,
            )
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config,
            )
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer(
                "viewfreq", torch.FloatTensor([(2**i) for i in range(viewbase_pe)])
            )
            dim0 = 3 + 3 * viewbase_pe * 2
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim - 3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width),
                nn.ReLU(inplace=True),
                *[
                    nn.Sequential(
                        nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)
                    )
                    for _ in range(rgbnet_depth - 2)
                ],
                nn.Linear(rgbnet_width, 3),
            ).cuda()
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print("dvgo: feature voxel grid", self.k0)
            print("dvgo: mlp", self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size

        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                path=mask_cache_path, mask_cache_thres=mask_cache_thres
            ).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]
                    ),
                    torch.linspace(
                        self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]
                    ),
                    torch.linspace(
                        self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]
                    ),
                ),
                -1,
            )
            mask = mask_cache(self_grid_xyz.cuda())
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)

        self.mask_cache = grid.MaskGrid(
            path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max
        )

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print("dvgo: voxel_size      ", self.voxel_size)
        print("dvgo: world_size      ", self.world_size)
        print("dvgo: voxel_size_base ", self.voxel_size_base)
        print("dvgo: voxel_size_ratio", self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            "xyz_min": self.xyz_min.cpu().numpy(),
            "xyz_max": self.xyz_max.cpu().numpy(),
            "num_voxels": self.num_voxels,
            "num_voxels_base": self.num_voxels_base,
            "alpha_init": self.alpha_init,
            "voxel_size_ratio": self.voxel_size_ratio,
            "mask_cache_path": self.mask_cache_path,
            "mask_cache_thres": self.mask_cache_thres,
            "mask_cache_world_size": list(self.mask_cache.mask.shape),
            "fast_color_thres": self.fast_color_thres,
            "density_type": self.density_type,
            "k0_type": self.k0_type,
            "density_config": self.density_config,
            "k0_config": self.k0_config,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ),
            -1,
        )
        nearest_dist = torch.stack(
            [
                (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
                for co in cam_o  # for memory saving
            ]
        ).amin(0)
        self.density.grid[nearest_dist[None, None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print("dvgo: scale_volume_grid start")
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print(
            "\x1b[6;30;42m"
            + "dvgo: scale_volume_grid scale world_size from"
            + "\x1b[0m",
            ori_world_size.tolist(),
            "to",
            self.world_size.tolist(),
        )

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        self.xyz_min[0], self.xyz_max[0], self.world_size[0]
                    ),
                    torch.linspace(
                        self.xyz_min[1], self.xyz_max[1], self.world_size[1]
                    ),
                    torch.linspace(
                        self.xyz_min[2], self.xyz_max[2], self.world_size[2]
                    ),
                ),
                -1,
            )
            self_alpha = F.max_pool3d(
                self.activate_density(self.density.get_dense_grid()),
                kernel_size=3,
                padding=1,
                stride=1,
            )[0, 0]
            self.mask_cache = grid.MaskGrid(
                path=None,
                mask=self.mask_cache(self_grid_xyz)
                & (self_alpha > self.fast_color_thres),
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
            )

        print("dvgo: scale_volume_grid finish")

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]
                ),
                torch.linspace(
                    self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]
                ),
                torch.linspace(
                    self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]
                ),
            ),
            -1,
        ).cuda()
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(
            cache_grid_alpha, kernel_size=3, padding=1, stride=1
        )[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres

    def voxel_count_views(
        self,
        rays_o_tr,
        rays_d_tr,
        imsz,
        near,
        far,
        stepsize,
        downrate=1,
        irregular_shape=False,
    ):
        print("dvgo: voxel_count_views start")
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()
        N_samples = (
            int(np.linalg.norm(np.array(self.world_size.cpu()) + 1) / stepsize) + 1
        )
        rng = torch.arange(N_samples)[None].float().cuda()
        count = torch.zeros_like(self.density.get_dense_grid()).cuda()
        device = rng.device

        for rays_o_, rays_d_ in zip(rays_o_tr, rays_d_tr):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = (
                    torch.from_numpy(rays_o_[::downrate, ::downrate])
                    .to(device)
                    .flatten(0, -2)
                    .split(10000)
                )
                rays_d_ = (
                    torch.from_numpy(rays_d_[::downrate, ::downrate])
                    .to(device)
                    .flatten(0, -2)
                    .split(10000)
                )

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max.cuda() - rays_o) / vec
                rate_b = (self.xyz_min.cuda() - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size.cuda() * rng
                interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
                rays_pts = (
                    rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                )
                ones(rays_pts).sum().backward()

            with torch.no_grad():
                count += ones.grid.grad > 1

        eps_time = time.time() - eps_time
        print("dvgo: voxel_count_views finish (eps time:", eps_time, "sec)")
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(
            shape
        )

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Check whether the rays hit the solved coarse geometry or not"""
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        from src.model.dvgo.__global__ import render_utils_cuda

        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
            rays_o,
            rays_d,
            self.xyz_min.cuda(),
            self.xyz_max.cuda(),
            near,
            far,
            stepdist,
        )[:3]

        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size

        from src.model.dvgo.__global__ import render_utils_cuda

        (
            ray_pts,
            mask_outbbox,
            ray_id,
            step_id,
            N_steps,
            t_min,
            t_max,
        ) = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist
        )
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert (
            len(rays_o.shape) == 2 and rays_o.shape[-1] == 3
        ), "Only suuport point queries in [N, 3] format"

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, **render_kwargs
        )
        interval = render_kwargs["stepsize"] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for color
        if self.rgbnet_full_implicit:
            pass
        else:
            k0 = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq.cuda()).flatten(-2)
            viewdirs_emb = torch.cat(
                [viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1
            )
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3]).cuda(),
            reduce="sum",
        )
        rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs["bg"]
        ret_dict.update(
            {
                "alphainv_last": alphainv_last,
                "weights": weights,
                "rgb_marched": rgb_marched,
                "raw_alpha": alpha,
                "raw_rgb": rgb,
                "ray_id": ray_id,
            }
        )

        if render_kwargs.get("render_depth", False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N]).cuda(),
                    reduce="sum",
                )
            ret_dict.update({"depth": depth})

        return ret_dict


""" Misc
"""


class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        """
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        """
        from src.model.dvgo.__global__ import render_utils_cuda

        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        """
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        """
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        from src.model.dvgo.__global__ import render_utils_cuda

        return (
            render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval),
            None,
            None,
        )


class Raw2Alpha_nonuni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        from src.model.dvgo.__global__ import render_utils_cuda

        exp, alpha = render_utils_cuda.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        from src.model.dvgo.__global__ import render_utils_cuda

        return (
            render_utils_cuda.raw2alpha_nonuni_backward(
                exp, grad_back.contiguous(), interval
            ),
            None,
            None,
        )


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        from src.model.dvgo.__global__ import render_utils_cuda

        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(
            alpha, ray_id.to(device=alpha.device), N
        )
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        from src.model.dvgo.__global__ import render_utils_cuda

        grad = render_utils_cuda.alpha2weight_backward(
            alpha,
            weights,
            T,
            alphainv_last,
            i_start,
            i_end,
            ctx.n_rays,
            grad_weights,
            grad_last,
        )
        return grad, None, None

# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Plenoxels (https://github.com/sxyu/svox2)
# Copyright (c) 2022 the Plenoxel authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

import src.model.plenoxel.autograd as autograd
import src.model.plenoxel.dataclass as dataclass
import src.model.plenoxel.utils as utils
from src.model.plenoxel.__global__ import (
    BASIS_TYPE_3D_TEXTURE,
    BASIS_TYPE_MLP,
    BASIS_TYPE_SH,
    _get_c_extension,
)

_C = _get_c_extension()


class SparseGrid(nn.Module):
    def __init__(
        self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        basis_type: int = BASIS_TYPE_SH,
        basis_dim: int = 9,
        use_z_order: bool = False,
        use_sphere_bound: bool = False,
        mlp_posenc_size: int = 0,
        mlp_width: int = 16,
        background_nlayers: int = 0,
        background_reso: int = 256,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.basis_type = basis_type
        if basis_type == BASIS_TYPE_SH:
            assert (
                utils.isqrt(basis_dim) is not None
            ), "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS
        ), f"basis_dim 1-{utils.MAX_SH_BASIS} supported"
        self.basis_dim = basis_dim

        self.mlp_posenc_size = mlp_posenc_size
        self.mlp_width = mlp_width

        self.background_nlayers = background_nlayers
        assert (
            background_nlayers == 0 or background_nlayers > 1
        ), "Please use at least 2 MSI layers (trilerp limitation)"
        self.background_reso = background_reso

        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert (
                len(reso) == 3
            ), "reso must be an integer or indexable object of 3 ints"

        if use_z_order and not (
            reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])
        ):
            use_z_order = False

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device="cpu", dtype=torch.float32)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device="cpu")
        if isinstance(center, torch.Tensor):
            center = center.to(device="cpu", dtype=torch.float32)
        else:
            center = torch.tensor(center, dtype=torch.float32, device="cpu")

        self.radius: torch.Tensor = radius  # CPU
        self.center: torch.Tensor = center  # CPU
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius

        n3: int = reduce(lambda x, y: x * y, reso)
        if use_z_order:
            init_links = utils.gen_morton(
                reso[0], device=device, dtype=torch.int32
            ).flatten()
        else:
            init_links = torch.arange(n3, device=device, dtype=torch.int32)

        if use_sphere_bound:
            X = torch.arange(reso[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(reso[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(reso[2], dtype=torch.float32, device=device) - 0.5
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            points = torch.addcmul(
                roffset.to(device=points.device),
                points,
                rscaling.to(device=points.device),
            )

            norms = points.norm(dim=-1)
            mask = norms <= 1.0 + (3**0.5) / gsz.max()
            self.capacity: int = mask.sum()

            data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
            idxs = init_links[mask].long()
            data_mask[idxs] = 1
            data_mask = torch.cumsum(data_mask, dim=0) - 1

            init_links[mask] = data_mask[idxs].int()
            init_links[~mask] = -1
        else:
            self.capacity = n3

        self.register_parameter(
            "density_data",
            nn.Parameter(
                torch.zeros(self.capacity, 1, dtype=torch.float32, device=device),
                requires_grad=True,
            ),
        )

        self.density_data.grad = torch.zeros_like(self.density_data)
        # Called sh for legacy reasons, but it's just the coeffients for whatever
        # spherical basis functions
        self.register_parameter(
            "sh_data",
            nn.Parameter(
                torch.zeros(
                    self.capacity,
                    self.basis_dim * 3,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=True,
            ),
        )
        self.sh_data.grad = torch.zeros_like(self.sh_data)

        self.register_parameter(
            "basis_data",
            nn.Parameter(
                torch.empty(0, 0, 0, 0, dtype=torch.float32, device=device),
                requires_grad=False,
            ),
        )

        self.background_links: Optional[torch.Tensor]
        self.background_data: Optional[torch.Tensor]
        if self.use_background:
            background_capacity = (self.background_reso**2) * 2
            background_links = torch.arange(
                background_capacity, dtype=torch.int32, device=device
            ).reshape(self.background_reso * 2, self.background_reso)
            self.register_buffer("background_links", background_links)
            self.register_parameter(
                "background_data",
                nn.Parameter(
                    torch.zeros(
                        background_capacity,
                        self.background_nlayers,
                        4,
                        dtype=torch.float32,
                        device=device,
                    ),
                    requires_grad=True,
                ),
            )
            self.background_data.grad = torch.zeros_like(self.background_data)
        else:
            self.register_parameter(
                "background_data",
                nn.Parameter(
                    torch.empty(0, 0, 0, dtype=torch.float32, device=device),
                    requires_grad=False,
                ),
            )

        self.register_buffer("links", init_links.view(reso))
        self.links: torch.Tensor
        self.opt = dataclass.RenderOptions()
        self.sparse_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_sh_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_background_indexer: Optional[torch.Tensor] = None
        self.density_rms: Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None
        self.background_rms: Optional[torch.Tensor] = None
        self.basis_rms: Optional[torch.Tensor] = None

        if self.links.is_cuda and use_sphere_bound:
            self.accelerate()

    @property
    def data_dim(self):
        """
        Get the number of channels in the data, including color + density
        (similar to svox 1)
        """
        return self.sh_data.size(1) + 1

    @property
    def use_background(self):
        return self.background_nlayers > 0

    @property
    def shape(self):
        return list(self.links.shape) + [self.data_dim]

    def _fetch_links(self, links):
        results_sigma = torch.zeros(
            (links.size(0), 1), device=links.device, dtype=torch.float32
        )
        results_sh = torch.zeros(
            (links.size(0), self.sh_data.size(1)),
            device=links.device,
            dtype=torch.float32,
        )
        mask = links >= 0
        idxs = links[mask].long()
        results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]
        return results_sigma, results_sh

    def sample(
        self,
        points: torch.Tensor,
        use_kernel: bool = True,
        grid_coords: bool = False,
        want_colors: bool = True,
    ):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates ([-0.5, reso[i]-0.5 ] in each dimension);
            more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
            else returns density and a dummy tensor to be ignored
            (much faster)

        :return: (density, color)
        """
        if use_kernel and self.links.is_cuda and _C is not None:
            assert points.is_cuda
            return autograd._SampleGridAutogradFunction.apply(
                self.density_data,
                self.sh_data,
                self._to_cpp(grid_coords=grid_coords),
                points,
                want_colors,
            )
        else:
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i) - 1)
            l = points.to(torch.long)
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i) - 2)
            wb = points - l
            wa = 1.0 - wb

            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])

            return samples_sigma, samples_rgb

    def forward(self, points: torch.Tensor, use_kernel: bool = True):
        return self.sample(points, use_kernel=use_kernel)

    def volume_render(
        self,
        rays: dataclass.Rays,
        use_kernel: bool = True,
        randomize: bool = False,
        return_raylen: bool = False,
    ):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :param return_raylen: bool, if true then only returns the length of the
                                    ray-cube intersection and quits
        :return: (N, 3), predicted RGB
        """
        assert rays.is_cuda
        basis_data = None
        return autograd._VolumeRenderFunction.apply(
            self.density_data,
            self.sh_data,
            basis_data,
            self.background_data if self.use_background else None,
            self._to_cpp(replace_basis_data=basis_data),
            rays._to_cpp(),
            self.opt._to_cpp(randomize=randomize),
            self.opt.backend,
        )

    def volume_render_fused(
        self,
        rays: dataclass.Rays,
        rgb_gt: torch.Tensor,
        randomize: bool = False,
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0,
    ):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Will update the *.grad tensors for each parameter
        You can then subtract the grad manually or use the optim_*_step methods

        See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param randomize: bool, whether to enable randomness
        :param beta_loss: float, weighting for beta loss to add to the gradient.
            (fused into the backward pass).
            This is average voer the rays in the batch.
            Beta loss also from neural volumes:
            [Lombardi et al., ToG 2019]
        :return: (N, 3), predicted RGB
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for fused"
        assert rays.is_cuda
        grad_density, grad_sh, grad_basis, grad_bg = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt, dtype=torch.float32)
        basis_data: Optional[torch.Tensor] = None
        self.sparse_grad_indexer = torch.zeros(
            (self.density_data.size(0),),
            dtype=torch.bool,
            device=self.density_data.device,
        )

        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh
        if self.basis_type != BASIS_TYPE_SH:
            grad_holder.grad_basis_out = grad_basis
        grad_holder.mask_out = self.sparse_grad_indexer
        if self.use_background:
            grad_holder.grad_background_out = grad_bg
            self.sparse_background_indexer = torch.zeros(
                list(self.background_data.shape[:-1]),
                dtype=torch.bool,
                device=self.background_data.device,
            )
            grad_holder.mask_background_out = self.sparse_background_indexer

        cu_fn = _C.__dict__[f"volume_render_{self.opt.backend}_fused"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            self._to_cpp(replace_basis_data=basis_data),
            rays._to_cpp(),
            self.opt._to_cpp(randomize=randomize),
            rgb_gt,
            beta_loss,
            sparsity_loss,
            rgb_out,
            grad_holder,
        )
        if self.basis_type == BASIS_TYPE_MLP:
            # Manually trigger MLP backward!
            basis_data.backward(grad_basis)

        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
        return rgb_out

    def volume_render_depth(
        self,
        rays: dataclass.Rays,
        sigma_thresh,
    ):
        """
        Volumetric depth rendering for rays

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param sigma_thresh: finds the first point where sigma strictly exceeds sigma_thresh

        :return: (N,)
        """
        assert not sigma_thresh is None
        return _C.volume_render_sigma_thresh(
            self._to_cpp(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            sigma_thresh,
        )

    def resample(
        self,
        reso: Union[int, List[int]],
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[dataclass.Camera]] = None,
        use_z_order: bool = False,
        accelerate: bool = True,
        weight_render_stop_thresh: float = 0.2,  # SHOOT, forgot to turn this off for main exps..
        max_elements: int = 0,
    ):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
            (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
            to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
            0.0 = no thresholding, 1.0 = hides everything.
            Useful for force-cutting off junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
            upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            if use_z_order and not (
                reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])
            ):
                use_z_order = False

            self.capacity: int = reduce(lambda x, y: x * y, reso)
            curr_reso = self.links.shape
            dtype = torch.float32
            reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]
            X = torch.linspace(
                reso_facts[0] - 0.5,
                curr_reso[0] - reso_facts[0] - 0.5,
                reso[0],
                dtype=dtype,
            )
            Y = torch.linspace(
                reso_facts[1] - 0.5,
                curr_reso[1] - reso_facts[1] - 0.5,
                reso[1],
                dtype=dtype,
            )
            Z = torch.linspace(
                reso_facts[2] - 0.5,
                curr_reso[2] - reso_facts[2] - 0.5,
                reso[2],
                dtype=dtype,
            )
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points.to(device=device)

            use_weight_thresh = cameras is not None

            batch_size = 720720
            all_sample_vals_density = []
            print("Pass 1/2 (density)")
            for i in range(0, len(points), batch_size):
                sample_vals_density, _ = self.sample(
                    points[i : i + batch_size], grid_coords=True, want_colors=False
                )
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)
            self.density_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None

            sample_vals_density = torch.cat(all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(reso)
                offset = (self._offset * gsz - 0.5).to(device=device)
                scaling = (self._scaling * gsz).to(device=device)
                max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                print(" Grid weight render", sample_vals_density.shape)
                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(
                        sample_vals_density,
                        cam._to_cpp(),
                        0.5,
                        weight_render_stop_thresh,
                        False,
                        offset,
                        scaling,
                        max_wt_grid,
                    )

                sample_vals_mask = max_wt_grid >= weight_thresh
                if (
                    max_elements > 0
                    and max_elements < max_wt_grid.numel()
                    and max_elements < torch.count_nonzero(sample_vals_mask)
                ):
                    # To bound the memory usage
                    weight_thresh_bounded = (
                        torch.topk(max_wt_grid.view(-1), k=max_elements, sorted=False)
                        .values.min()
                        .item()
                    )
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    print(" Readjusted weight thresh to fit to memory:", weight_thresh)
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if (
                    max_elements > 0
                    and max_elements < sample_vals_density.numel()
                    and max_elements < torch.count_nonzero(sample_vals_mask)
                ):
                    # To bound the memory usage
                    sigma_thresh_bounded = (
                        torch.topk(
                            sample_vals_density.view(-1), k=max_elements, sorted=False
                        )
                        .values.min()
                        .item()
                    )
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    print(" Readjusted sigma thresh to fit to memory:", sigma_thresh)
                    sample_vals_mask = sample_vals_density >= sigma_thresh

                if self.opt.last_sample_opaque:
                    # Don't delete the last z layer
                    sample_vals_mask[:, :, -1] = 1

            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()

            # Now we can get the colors for the sparse points
            points = points[sample_vals_mask]
            print("Pass 2/2 (color), eval", cnz, "sparse pts")
            all_sample_vals_sh = []
            for i in range(0, len(points), batch_size):
                _, sample_vals_sh = self.sample(
                    points[i : i + batch_size], grid_coords=True, want_colors=True
                )
                all_sample_vals_sh.append(sample_vals_sh)

            sample_vals_sh = (
                torch.cat(all_sample_vals_sh, dim=0)
                if len(all_sample_vals_sh)
                else torch.empty_like(self.sh_data[:0])
            )
            del self.density_data
            del self.sh_data
            del all_sample_vals_sh

            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full(
                    (sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32
                )
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = (
                    torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
                )
                init_links[~sample_vals_mask] = -1

            self.capacity = cnz
            print(" New cap:", self.capacity)
            del sample_vals_mask
            print("density", sample_vals_density.shape, sample_vals_density.dtype)
            print("sh", sample_vals_sh.shape, sample_vals_sh.dtype)
            print("links", init_links.shape, init_links.dtype)
            self.density_data = nn.Parameter(
                sample_vals_density.view(-1, 1).to(device=device)
            )
            self.sh_data = nn.Parameter(sample_vals_sh.to(device=device))
            self.links = init_links.view(reso).to(device=device)

            if accelerate and self.links.is_cuda:
                self.accelerate()

    def sparsify_background(
        self, sigma_thresh: float = 1.0, dilate: int = 1  # BEFORE resampling!
    ):
        device = self.background_links.device
        sigma_mask = torch.zeros(
            list(self.background_links.shape) + [self.background_nlayers],
            dtype=torch.bool,
            device=device,
        ).view(-1, self.background_nlayers)
        nonempty_mask = self.background_links.view(-1) >= 0
        data_mask = self.background_data[..., -1] >= sigma_thresh
        sigma_mask[nonempty_mask] = data_mask
        sigma_mask = sigma_mask.view(
            list(self.background_links.shape) + [self.background_nlayers]
        )
        for _ in range(int(dilate)):
            sigma_mask = _C.dilate(sigma_mask)

        sigma_mask = sigma_mask.any(-1) & nonempty_mask.view(
            self.background_links.shape
        )
        self.background_links[~sigma_mask] = -1
        retain_vals = self.background_links[sigma_mask]
        self.background_links[sigma_mask] = torch.arange(
            retain_vals.size(0), dtype=torch.int32, device=device
        )
        self.background_data = nn.Parameter(
            self.background_data.data[retain_vals.long()]
        )

    def resize(self, basis_dim: int):
        """
        Modify the size of the data stored in the voxels. Called expand/shrink in svox 1.

        :param basis_dim: new basis dimension, must be square number
        """
        assert (
            utils.isqrt(basis_dim) is not None
        ), "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS
        ), f"basis_dim 1-{utils.MAX_SH_BASIS} supported"
        old_basis_dim = self.basis_dim
        self.basis_dim = basis_dim
        device = self.sh_data.device
        old_data = self.sh_data.data.cpu()

        shrinking = basis_dim < old_basis_dim
        sigma_arr = torch.tensor([0])
        if shrinking:
            shift = old_basis_dim
            arr = torch.arange(basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])
        else:
            shift = basis_dim
            arr = torch.arange(old_basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])

        del self.sh_data
        new_data = torch.zeros((old_data.size(0), 3 * basis_dim + 1), device="cpu")
        if shrinking:
            new_data[:] = old_data[..., remap]
        else:
            new_data[..., remap] = old_data
        new_data = new_data.to(device=device)
        self.sh_data = nn.Parameter(new_data)
        self.sh_rms = None

    def accelerate(self):
        """
        Accelerate
        """
        assert (
            _C is not None and self.links.is_cuda
        ), "CUDA extension is currently required for accelerate"
        _C.accel_dist_prop(self.links)

    def world2grid(self, points):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        offset = self._offset * gsz - 0.5
        scaling = self._scaling * gsz
        return torch.addcmul(
            offset.to(device=points.device), points, scaling.to(device=points.device)
        )

    def grid2world(self, points):
        """
        Grid coordinates to world coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        roffset = self.radius * (1.0 / gsz - 1.0) + self.center
        rscaling = 2.0 * self.radius / gsz
        return torch.addcmul(
            roffset.to(device=points.device), points, rscaling.to(device=points.device)
        )

    def tv(
        self,
        logalpha: bool = False,
        logalpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
    ):
        """
        Compute total variation over sigma,
        similar to Neural Volumes [Lombardi et al., ToG 2019]

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
        mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        return autograd._TotalVariationFunction.apply(
            self.density_data,
            self.links,
            0,
            1,
            logalpha,
            logalpha_delta,
            False,
            ndc_coeffs,
        )

    def tv_color(
        self,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        logalpha: bool = False,
        logalpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
    ):
        """
        Compute total variation on color

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
        Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
        Default None = all dimensions until the end.

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
        mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        return autograd._TotalVariationFunction.apply(
            self.sh_data,
            self.links,
            start_dim,
            end_dim,
            logalpha,
            logalpha_delta,
            True,
            ndc_coeffs,
        )

    def inplace_tv_grad(
        self,
        grad: torch.Tensor,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool = False,
        logalpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True,
    ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.density_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        assert not logalpha, "No longer supported"
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.tv_grad_sparse(
                    self.links,
                    self.density_data,
                    rand_cells,
                    self._get_sparse_grad_indexer(),
                    0,
                    1,
                    scaling,
                    logalpha,
                    logalpha_delta,
                    False,
                    self.opt.last_sample_opaque,
                    ndc_coeffs[0],
                    ndc_coeffs[1],
                    grad,
                )
        else:
            _C.tv_grad(
                self.links,
                self.density_data,
                0,
                1,
                scaling,
                logalpha,
                logalpha_delta,
                False,
                ndc_coeffs[0],
                ndc_coeffs[1],
                grad,
            )
            self.sparse_grad_indexer: Optional[torch.Tensor] = None

    def inplace_tv_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool = False,
        logalpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True,
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
            Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
            Default None = all dimensions until the end.
        """
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_sh_grad_indexer()
                #  with utils.Timing("actual_tv_color"):
                _C.tv_grad_sparse(
                    self.links,
                    self.sh_data,
                    rand_cells,
                    indexer,
                    start_dim,
                    end_dim,
                    scaling,
                    logalpha,
                    logalpha_delta,
                    True,
                    False,
                    ndc_coeffs[0],
                    ndc_coeffs[1],
                    grad,
                )
        else:
            _C.tv_grad(
                self.links,
                self.sh_data,
                start_dim,
                end_dim,
                scaling,
                logalpha,
                logalpha_delta,
                True,
                ndc_coeffs[0],
                ndc_coeffs[1],
                grad,
            )
            self.sparse_sh_grad_indexer = None

    def inplace_tv_lumisphere_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool = False,
        logalpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        dir_factor: float = 1.0,
        dir_perturb_radians: float = 0.05,
    ):
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert self.basis_type != BASIS_TYPE_MLP, "MLP not supported"
        rand_cells = self._get_rand_cells(sparse_frac)
        grad_holder = _C.GridOutputGrads()

        indexer = self._get_sparse_sh_grad_indexer()
        assert indexer is not None
        grad_holder.mask_out = indexer
        grad_holder.grad_sh_out = grad

        batch_size = rand_cells.size(0)

        dirs = torch.randn(3, device=rand_cells.device)
        dirs /= torch.norm(dirs)

        sh_mult = utils.eval_sh_bases(self.basis_dim, dirs[None])
        sh_mult = sh_mult[0]

        if dir_factor > 0.0:
            axis = torch.randn((batch_size, 3))
            axis /= torch.norm(axis, dim=-1, keepdim=True)
            axis *= dir_perturb_radians
            R = Rotation.from_rotvec(axis.numpy()).as_matrix()
            R = torch.from_numpy(R).float().to(device=rand_cells.device)
            dirs_perturb = (R * dirs.unsqueeze(-2)).sum(-1)
        else:
            dirs_perturb = dirs  # Dummy, since it won't be used

        sh_mult_u = utils.eval_sh_bases(self.basis_dim, dirs_perturb[None])
        sh_mult_u = sh_mult_u[0]

        _C.lumisphere_tv_grad_sparse(
            self._to_cpp(),
            rand_cells,
            sh_mult,
            sh_mult_u,
            scaling,
            ndc_coeffs[0],
            ndc_coeffs[1],
            dir_factor,
            grad_holder,
        )

    def inplace_l2_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
    ):
        """
        Add gradient of L2 regularization for color
        directly into the gradient tensor, multiplied by 'scaling'
        (no CUDA extension used)

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
            Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
            Default None = all dimensions until the end.
        """
        with torch.no_grad():
            if end_dim is None:
                end_dim = self.sh_data.size(1)
            end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
            start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

            if self.sparse_sh_grad_indexer is None:
                scale = scaling / self.sh_data.size(0)
                grad[:, start_dim:end_dim] += scale * self.sh_data[:, start_dim:end_dim]
            else:
                indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
                nz: int = (
                    torch.count_nonzero(indexer).item()
                    if indexer.dtype == torch.bool
                    else indexer.size(0)
                )
                scale = scaling / nz
                grad[indexer, start_dim:end_dim] += (
                    scale * self.sh_data[indexer, start_dim:end_dim]
                )

    def inplace_tv_background_grad(
        self,
        grad: torch.Tensor,
        scaling: float = 1.0,
        scaling_density: Optional[float] = None,
        sparse_frac: float = 0.01,
        contiguous: bool = False,
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        rand_cells_bg = self._get_rand_cells_background(sparse_frac, contiguous)
        indexer = self._get_sparse_background_grad_indexer()
        if scaling_density is None:
            scaling_density = scaling
        _C.msi_tv_grad_sparse(
            self.background_links,
            self.background_data,
            rand_cells_bg,
            indexer,
            scaling,
            scaling_density,
            grad,
        )

    def optim_density_step(
        self,
        lr: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop or sgd step on density
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer()
        if optim == "rmsprop":
            if (
                self.density_rms is None
                or self.density_rms.shape != self.density_data.shape
            ):
                del self.density_rms
                self.density_rms = torch.zeros_like(
                    self.density_data.data
                )  # FIXME init?
            _C.rmsprop_step(
                self.density_data.data,
                self.density_rms,
                self.density_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr,
            )
        elif optim == "sgd":
            _C.sgd_step(self.density_data.data, self.density_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")

    def optim_sh_step(
        self,
        lr: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop/SGD step on SH
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
        if optim == "rmsprop":
            if self.sh_rms is None or self.sh_rms.shape != self.sh_data.shape:
                del self.sh_rms
                self.sh_rms = torch.zeros_like(self.sh_data.data)  # FIXME init?
            _C.rmsprop_step(
                self.sh_data.data,
                self.sh_rms,
                self.sh_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr,
            )
        elif optim == "sgd":
            _C.sgd_step(self.sh_data.data, self.sh_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")

    def optim_background_step(
        self,
        lr_sigma: float,
        lr_color: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop or sgd step on density
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(bg=True)
        n_chnl = self.background_data.size(-1)
        if optim == "rmsprop":
            if (
                self.background_rms is None
                or self.background_rms.shape != self.background_data.shape
            ):
                del self.background_rms
                self.background_rms = torch.zeros_like(
                    self.background_data.data
                )  # FIXME init?
            _C.rmsprop_step(
                self.background_data.data.view(-1, n_chnl),
                self.background_rms.view(-1, n_chnl),
                self.background_data.grad.view(-1, n_chnl),
                indexer,
                beta,
                lr_color,
                epsilon,
                -1e9,
                lr_sigma,
            )
        elif optim == "sgd":
            _C.sgd_step(
                self.background_data.data.view(-1, n_chnl),
                self.background_data.grad.view(-1, n_chnl),
                indexer,
                lr_color,
                lr_sigma,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")

    def optim_basis_step(
        self,
        lr: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop/SGD step on SH
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        if optim == "rmsprop":
            if self.basis_rms is None or self.basis_rms.shape != self.basis_data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(self.basis_data.data)
            self.basis_rms.mul_(beta).addcmul_(
                self.basis_data.grad, self.basis_data.grad, value=1.0 - beta
            )
            denom = self.basis_rms.sqrt().add_(epsilon)
            self.basis_data.data.addcdiv_(self.basis_data.grad, denom, value=-lr)
        elif optim == "sgd":
            self.basis_data.grad.mul_(lr)
            self.basis_data.data -= self.basis_data.grad
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")
        self.basis_data.grad.zero_()

    @property
    def basis_type_name(self):
        if self.basis_type == BASIS_TYPE_SH:
            return "SH"
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            return "3D_TEXTURE"
        elif self.basis_type == BASIS_TYPE_MLP:
            return "MLP"
        return "UNKNOWN"

    def __repr__(self):
        return (
            f"svox2.SparseGrid(basis_type={self.basis_type_name}, "
            + f"basis_dim={self.basis_dim}, "
            + f"reso={list(self.links.shape)}, "
            + f"capacity:{self.sh_data.size(0)})"
        )

    def is_cubic_pow2(self):
        """
        Check if the current grid is cubic (same in all dims) with power-of-2 size.
        This allows for conversion to svox 1 and Z-order curve (Morton code)
        """
        reso = self.links.shape
        return reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])

    def _to_cpp(
        self,
        grid_coords: bool = False,
        replace_basis_data: Optional[torch.Tensor] = None,
    ):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.sh_data = self.sh_data
        gspec.links = self.links
        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._offset)
        else:
            gsz = self._grid_size()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz

        gspec.basis_dim = self.basis_dim
        gspec.basis_type = self.basis_type
        if replace_basis_data:
            gspec.basis_data = replace_basis_data
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            gspec.basis_data = self.basis_data

        if self.use_background:
            gspec.background_links = self.background_links
            gspec.background_data = self.background_data
        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device="cpu", dtype=torch.float32)

    def _get_data_grads(self):
        ret = []
        for subitem in ["density_data", "sh_data", "basis_data", "background_data"]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                    not hasattr(param, "grad")
                    or param.grad is None
                    or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = torch.empty(
                (0,), dtype=torch.bool, device=self.density_data.device
            )
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = torch.empty(
                (0,), dtype=torch.bool, device=self.density_data.device
            )
        return indexer

    def _get_sparse_background_grad_indexer(self):
        indexer = self.sparse_background_indexer
        if indexer is None:
            indexer = torch.empty(
                (0, 0, 0, 0), dtype=torch.bool, device=self.density_data.device
            )
        return indexer

    def _maybe_convert_sparse_grad_indexer(self, sh=False, bg=False):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        indexer = self.sparse_sh_grad_indexer if sh else self.sparse_grad_indexer
        if bg:
            indexer = self.sparse_background_indexer
            if indexer is not None:
                indexer = indexer.view(-1)
        if indexer is None:
            return torch.empty((), device=self.density_data.device)
        if (
            indexer.dtype == torch.bool
            and torch.count_nonzero(indexer).item() < indexer.size(0) // 8
        ):
            # Highly sparse (use index)
            indexer = torch.nonzero(indexer.flatten(), as_tuple=False).flatten()
        return indexer

    def _get_rand_cells(
        self, sparse_frac: float, force: bool = False, contiguous: bool = True
    ):
        if sparse_frac < 1.0 or force:
            assert (
                self.sparse_grad_indexer is None
                or self.sparse_grad_indexer.dtype == torch.bool
            ), "please call sparse loss after rendering and before gradient updates"
            grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
            sparse_num = max(int(sparse_frac * grid_size), 1)
            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = torch.arange(
                    start,
                    start + sparse_num,
                    dtype=torch.int32,
                    device=self.links.device,
                )

                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start :] -= grid_size
                return arr
            else:
                return torch.randint(
                    0,
                    grid_size,
                    (sparse_num,),
                    dtype=torch.int32,
                    device=self.links.device,
                )
        return None

    def _get_rand_cells_background(self, sparse_frac: float, contiguous: bool = True):
        assert (
            self.use_background
        ), "Can only use sparse background loss if using background"
        assert (
            self.sparse_background_indexer is None
            or self.sparse_background_indexer.dtype == torch.bool
        ), "please call sparse loss after rendering and before gradient updates"
        grid_size = (
            self.background_links.size(0)
            * self.background_links.size(1)
            * self.background_data.size(1)
        )
        sparse_num = max(int(sparse_frac * grid_size), 1)
        if contiguous:
            start = np.random.randint(0, grid_size)  # - sparse_num + 1)
            arr = torch.arange(
                start, start + sparse_num, dtype=torch.int32, device=self.links.device
            )
            if start > grid_size - sparse_num:
                arr[grid_size - sparse_num - start :] -= grid_size
            return arr
        else:
            return torch.randint(
                0, grid_size, (sparse_num,), dtype=torch.int32, device=self.links.device
            )

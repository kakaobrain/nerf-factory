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
from random import random
from typing import List, Optional, Tuple, Union

import torch

import src.model.plenoxel.utils as utils
from src.model.plenoxel.__global__ import _get_c_extension

_C = _get_c_extension()


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    def __init__(
        self,
        backend: str = "cuvol",
        background_brightness: float = 1.0,
        step_size: float = 0.5,
        sigma_thresh: float = 1e-10,
        stop_thresh: float = 1e-7,
        last_sample_opaque: bool = False,
        near_clip: float = 0.0,
        use_spheric_clip: bool = False,
    ):
        self.backend = backend
        self.background_brightness = background_brightness
        self.step_size = step_size
        self.sigma_thresh = sigma_thresh
        self.stop_thresh = stop_thresh
        self.last_sample_opaque = last_sample_opaque
        self.near_clip = near_clip
        self.use_spheric_clip = use_spheric_clip

    def _to_cpp(self, randomize: bool = False):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip
        opt.last_sample_opaque = self.last_sample_opaque

        return opt


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w.float()
        spec.fx = float(self.fx_val)
        spec.fy = float(self.fy_val)
        spec.cx = float(self.cx_val)
        spec.cy = float(self.cy_val)
        spec.width = int(self.width)
        spec.height = int(self.height)
        spec.ndc_coeffx = float(self.ndc_coeffs[0])
        spec.ndc_coeffy = float(self.ndc_coeffs[1])
        return spec

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda

# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import numpy as np


def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]
    ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz
    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)

    return origins, directions


def batchified_get_rays(
    intrinsics,
    extrinsics,
    image_sizes,
    use_pixel_centers,
    get_radii,
    ndc_coord,
    ndc_coeffs,
    multlosses,
):

    radii = None
    multloss_expand = None

    center = 0.5 if use_pixel_centers else 0.0
    mesh_grids = [
        np.meshgrid(
            np.arange(w, dtype=np.float32) + center,
            np.arange(h, dtype=np.float32) + center,
            indexing="xy",
        )
        for (h, w) in image_sizes
    ]

    i_coords = [mesh_grid[0] for mesh_grid in mesh_grids]
    j_coords = [mesh_grid[1] for mesh_grid in mesh_grids]

    dirs = [
        np.stack(
            [
                (i - intrinsic[0][2]) / intrinsic[0][0],
                (j - intrinsic[1][2]) / intrinsic[1][1],
                np.ones_like(i),
            ],
            -1,
        )
        for (intrinsic, i, j) in zip(intrinsics, i_coords, j_coords)
    ]

    rays_o = np.concatenate(
        [
            np.tile(extrinsic[np.newaxis, :3, 3], (1, h * w, 1)).reshape(-1, 3)
            for (extrinsic, (h, w)) in zip(extrinsics, image_sizes)
        ]
    ).astype(np.float32)

    rays_d = np.concatenate(
        [
            np.einsum("hwc, rc -> hwr", dir, extrinsic[:3, :3]).reshape(-1, 3)
            for (dir, extrinsic) in zip(dirs, extrinsics)
        ]
    ).astype(np.float32)

    viewdirs = rays_d
    viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    if ndc_coord:
        rays_o, rays_d = convert_to_ndc(rays_o, rays_d, ndc_coeffs)

    if get_radii:

        if not ndc_coord:
            rays_d_orig = [
                np.einsum("hwc, rc -> hwr", dir, extrinsic[:3, :3])
                for (dir, extrinsic) in zip(dirs, extrinsics)
            ]
            dx = [
                np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1))
                for v in rays_d_orig
            ]
            dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
            _radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
            radii = np.concatenate([radii_each.reshape(-1) for radii_each in _radii])[
                ..., None
            ]

        else:
            rays_o_orig, cnt = [], 0
            for (h, w) in image_sizes:
                rays_o_orig.append(rays_o[cnt : cnt + h * w].reshape(h, w, 3))
                cnt += h * w

            dx = [
                np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1))
                for v in rays_o_orig
            ]
            dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
            dy = [
                np.sqrt(np.sum((v[:, :-1, :] - v[:, 1:, :]) ** 2, -1))
                for v in rays_o_orig
            ]
            dy = [np.concatenate([v, v[:, -2:-1]], 1) for v in dy]
            _radii = [(vx + vy)[..., None] / np.sqrt(12) for (vx, vy) in zip(dx, dy)]
            radii = np.concatenate([radii_each.reshape(-1) for radii_each in _radii])[
                ..., None
            ]

    if multlosses is not None:
        multloss_expand = np.concatenate(
            [
                np.array([scale] * (h * w))
                for (scale, (h, w)) in zip(multlosses, image_sizes)
            ]
        )[..., None]

    return rays_o, rays_d, viewdirs, radii, multloss_expand

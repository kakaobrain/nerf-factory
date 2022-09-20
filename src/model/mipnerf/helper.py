# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Mip-NeRF (https://github.com/google/mipnerf)
# Copyright (c) 2021 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import numpy as np
import torch


def img2mse(x, y, mask):
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * mask) / mask.sum()


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def sample_along_rays(
    rays_o,
    rays_d,
    radii,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    ray_shape,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    means, covs = cast_rays(t_vals, rays_o, rays_d, radii, ray_shape)

    return t_vals, (means, covs)


def resample_along_rays(
    rays_o,
    rays_d,
    radii,
    t_vals,
    weights,
    randomized,
    ray_shape,
    stop_level_grad,
    resample_padding,
):

    weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
    weights_max = torch.fmax(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    weights = weights_blur + resample_padding

    new_t_vals = sorted_piecewise_constant_pdf(
        t_vals, weights, t_vals.shape[-1], randomized
    )
    if stop_level_grad:
        new_t_vals = new_t_vals.detach()

    means, covs = cast_rays(new_t_vals, rays_o, rays_d, radii, ray_shape)

    return new_t_vals, (means, covs)


# 2**(-52) is the minimum epsilon value
def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], axis=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        axis=-1,
    )

    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples, device=weights.device) * s
        u += torch.rand_like(u) * (s - float_min_eps)
        u = torch.fmin(u, torch.ones_like(u) * (1.0 - float_min_eps))
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def integrated_pos_enc(samples, min_deg, max_deg):
    x, x_cov_diag = samples
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    shape = list(x.shape[:-1]) + [-1]
    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None] ** 2, shape)

    return expected_sin(
        torch.cat([y, y + 0.5 * np.pi], axis=-1), torch.cat([y_var] * 2, axis=-1)
    )[0]


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(
        -torch.cat(
            [
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], axis=-1),
            ],
            axis=-1,
        )
    )
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(axis=-2)
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) / acc
    distance = torch.clip(distance, t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, distance, acc, weights


def pos_enc(x, min_deg, max_deg, append_identity):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], axis=-1)
    else:
        return four_feat


def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2
    y_var = torch.fmax(torch.zeros_like(y_var), y_var)
    return y, y_var


def lift_gaussian(d, t_mean, t_var, r_var):

    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.sum(d**2, dim=-1, keepdim=True)
    thresholds = torch.ones_like(d_mag_sq) * 1e-10
    d_mag_sq = torch.fmax(d_mag_sq, thresholds)

    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag

    return mean, cov_diag


# According to the link below, numerically stable implementations are required.
# https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/mip.py#L88


def conical_frustum_to_gaussian(d, t0, t1, radius):

    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * (
        (hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2
    )
    r_var = radius**2 * (
        (mu**2) / 4
        + (5 / 12) * hw**2
        - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2)
    )

    return lift_gaussian(d, t_mean, t_var, r_var)


def cylinder_to_gaussian(d, t0, t1, radius):

    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0) ** 2 / 12

    return lift_gaussian(d, t_mean, t_var, r_var)


def cast_rays(t_vals, origins, directions, radii, ray_shape):
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == "cone":
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == "cylinder":
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii)
    means = means + origins[..., None, :]
    return means, covs

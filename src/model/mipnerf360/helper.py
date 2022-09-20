# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Mip-NeRF360 (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import itertools

import functorch
import numpy as np
import torch
import torch.nn.functional as F

eps = 1.1920929e-07

# Verified
def img2mse(x, y):
    return torch.mean((x - y) ** 2)


# Verified
def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


# Verified


def contract(mean, cov, is_train=True):

    bsz, num_samples, dim = mean.shape

    def _contract(x):
        x_mag_sq = torch.sum(x**2, dim=-1, keepdim=True).clip(min=1e-32)
        z = torch.where(
            x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x
        )
        return z

    mean_reshape = mean.reshape(bsz * num_samples, dim)
    cov_reshape = cov.reshape(bsz * num_samples, dim, dim)

    if is_train:
        ft_mean = functorch.vjp(_contract, mean)[0]
        ft_jacobian = functorch.vmap(functorch.jacrev(_contract, argnums=0))(
            mean_reshape
        )

    else:
        with torch.inference_mode(False):
            with torch.enable_grad():
                ft_mean = functorch.vjp(_contract, mean)[0]
                ft_jacobian = functorch.vmap(functorch.jacrev(_contract, argnums=0))(
                    mean_reshape
                )

    ft_cov = torch.einsum("bij, bjk -> bik", ft_jacobian, cov_reshape)
    ft_cov = torch.einsum("bij, bkj -> bik", ft_cov, ft_jacobian)

    return (
        ft_mean.reshape(bsz, num_samples, dim).detach(),
        ft_cov.reshape(bsz, num_samples, dim, dim).detach(),
    )


# Verified
def lift_and_diagonalize(means, covs, basis):
    fn_mean = means @ basis
    fn_cov_diag = torch.sum(basis[None, None, ...] * (covs @ basis), dim=-2)
    return fn_mean, fn_cov_diag


# Verified
def integrated_pos_enc(mean, var, min_deg, max_deg):
    scales = 2 ** torch.arange(min_deg, max_deg).type_as(mean)
    shape = list(mean.shape[:-1]) + [
        -1,
    ]
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None] ** 2, shape)

    return expected_sin(
        torch.cat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1),
        torch.cat([scaled_var] * 2, dim=-1),
    )


# Verified
def pos_enc(x, min_deg, max_deg, append_identity):
    scales = 2 ** torch.arange(min_deg, max_deg).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), x.shape[:-1] + (-1,))
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


# Verified
def expected_sin(mean, var):
    return torch.exp(-0.5 * var) * torch.sin(mean)


# Verified
def searchsorted(a, v):
    i = torch.arange(a.shape[-1], device=a.device)
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = torch.where(v_ge_a, i[..., :, None], i[..., :1, None]).max(dim=-2).values
    idx_hi = torch.where(~v_ge_a, i[..., :, None], i[..., -1:, None]).min(dim=-2).values
    return idx_lo, idx_hi


def inner_outer(t0, t1, y1):

    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = torch.take_along_dim(cy1, idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1, idx_hi, dim=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]
    y0_inner = torch.where(
        idx_hi[..., :-1] <= idx_lo[..., 1:],
        cy1_lo[..., 1:] - cy1_hi[..., :-1],
        torch.zeros_like(cy1_lo[..., 1:]),
    )

    return y0_inner, y0_outer


# Verified
def lossfun_outer(t, w, t_env, w_env):
    _, w_outer = inner_outer(t, t_env, w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)


# Verified
def lossfun_distortion(t, w):
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


# Verified
def max_dilate(t, w, dilation, domain):
    t0 = t[..., :-1] - dilation
    t1 = t[..., 1:] + dilation
    t_dilate = torch.sort(torch.cat([t, t0, t1], dim=-1), dim=-1).values
    t_dilate = torch.clip(t_dilate, domain[0], domain[1])
    mask = (t0[..., None, :] <= t_dilate[..., None]) & (
        t1[..., None, :] > t_dilate[..., None]
    )
    w_dilate = (
        torch.where(mask, w[..., None, :], torch.zeros_like(w[..., None, :]))
        .max(dim=-1)
        .values[..., :-1]
    )
    return t_dilate, w_dilate


def construct_ray_warps(t_near, t_far):
    s_near, s_far = 1 / t_near, 1 / t_far
    t_to_s = lambda t: (1 / t - s_near) / (s_far - s_near)
    s_to_t = lambda s: 1 / (s * s_far + (1 - s) * s_near)
    return t_to_s, s_to_t


# Verified
def weight_to_pdf(t, w):
    return w.squeeze(-1) / torch.clip(t[..., 1:] - t[..., :-1], min=eps)


# Verified
def pdf_to_weight(t, p):
    return p * (t[..., 1:] - t[..., :-1])


# Verified
def max_dilate_weights(t, w, dilation, domain, renormalize):
    p = weight_to_pdf(t, w)
    t_dilate, p_dilate = max_dilate(t, p, dilation, domain)
    w_dilate = pdf_to_weight(t_dilate, p_dilate)
    if renormalize:
        w_dilate /= torch.clip(torch.sum(w_dilate, dim=-1, keepdim=True), min=eps)
    return t_dilate, w_dilate


# Verified
def integrate_weights(w):
    cumsum = torch.cumsum(w[..., :-1], dim=-1)
    cw = cumsum.clip(max=1.0)
    shape = cw.shape[:-1] + (1,)
    cw0 = torch.cat(
        [torch.zeros(shape).type_as(cw), cw, torch.ones(shape).type_as(cw)], dim=-1
    )
    return cw0


# Verified
def sorted_interp(x, xp, fp):

    mask = x[..., None, :] >= xp[..., :, None]

    fp0 = torch.max(torch.where(mask, fp[..., None], fp[..., :1, None]), dim=-2).values
    fp1 = torch.min(
        torch.where(~mask, fp[..., None], fp[..., -1:, None]), dim=-2
    ).values
    xp0 = torch.max(torch.where(mask, xp[..., None], xp[..., :1, None]), dim=-2).values
    xp1 = torch.min(
        torch.where(~mask, xp[..., None], xp[..., -1:, None]), dim=-2
    ).values

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret


# Verified
def invert_cdf(u, t, w_logits):
    w = F.softmax(w_logits, dim=-1)
    cw = integrate_weights(w)
    t_new = sorted_interp(u, cw, t)
    return t_new


# Verified
def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    density_delta = density * delta

    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = torch.cat(
            [
                density_delta[..., :-1],
                torch.full_like(density_delta[..., -1:], torch.inf),
            ],
            dim=-1,
        )

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(
        -torch.cat(
            [
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], dim=-1),
            ],
            dim=-1,
        )
    )
    weights = alpha * trans
    return weights, alpha, trans


# Verified
def volumetric_rendering(
    rgbs, weights, tdist, bg_rgbs, t_far, compute_extras, extras=None
):
    rendering = {}

    acc = weights.sum(dim=-1)
    bg_w = torch.clip(1 - acc[..., None], min=0)  # The weight of the background.
    rgb = (weights[..., None] * rgbs).sum(dim=-2) + bg_w * bg_rgbs
    rendering["rgb"] = rgb

    return rendering


# Verified
def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == "cone":
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == "cylinder":
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


# Verified
def conical_frustum_to_gaussian(d, t0, t1, radius, diag):

    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2).clip(min=eps)
    denom = (3 * mu**2 + hw**2).clip(min=eps)
    t_var = (hw**2) / 3 - (4 / 15) * hw**4 * (12 * mu**2 - hw**2) / denom**2
    r_var = (mu**2) / 4 + (5 / 12) * hw**2 - (4 / 15) * (hw**4) / denom
    r_var *= radius**2

    return lift_gaussian(d, t_mean, t_var, r_var, diag)


# Verified
def cylinder_to_gaussian(d, t0, t1, radius, diag):

    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0) ** 2 / 12

    return lift_gaussian(d, t_mean, t_var, r_var, diag)


# Verified
def lift_gaussian(d, t_mean, t_var, r_var, diag):

    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.sum(d**2, dim=-1, keepdim=True)
    d_mag_sq = d_mag_sq.clip(min=1e-10)

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1]).type_as(d)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


# Verified
def sample(
    randomized,
    t,
    w_logits,
    num_samples,
    single_jitter=False,
    deterministic_center=False,
):
    if not randomized:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = torch.linspace(pad, 1 - pad - eps, num_samples)
        else:
            u = torch.linspace(0, 1 - eps, num_samples)
        u = torch.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = (
            torch.linspace(0, 1 - u_max, num_samples)
            + torch.rand(t.shape[:-1] + (d,)) * max_jitter
        )

    u = u.type_as(t)

    return invert_cdf(u, t, w_logits)


# Verified
def sample_intervals(
    randomized,
    t,
    w_logits,
    num_samples,
    single_jitter=False,
    domain=(-torch.inf, torch.inf),
):
    centers = sample(
        randomized,
        t,
        w_logits,
        num_samples,
        single_jitter,
        deterministic_center=True,
    )

    mid = (centers[..., 1:] + centers[..., :-1]) / 2
    min_val, max_val = domain
    first = torch.clip(2 * centers[..., :1] - mid[..., :1], min=min_val)
    last = torch.clip(2 * centers[..., -1:] - mid[..., -1:], max=max_val)

    t_samples = torch.cat([first, mid, last], dim=-1)
    return t_samples


## Geopoly


def compute_sq_dist(mat0, mat1=None):
    """Compute the squared Euclidean distance between all pairs of columns."""
    if mat1 is None:
        mat1 = mat0
    # Use the fact that ||x - y||^2 == ||x||^2 + ||y||^2 - 2 x^T y.
    sq_norm0 = np.sum(mat0**2, 0)
    sq_norm1 = np.sum(mat1**2, 0)
    sq_dist = sq_norm0[:, None] + sq_norm1[None, :] - 2 * mat0.T @ mat1
    sq_dist = np.maximum(0, sq_dist)  # Negative values must be numerical errors.
    return sq_dist


def compute_tesselation_weights(v):
    """Tesselate the vertices of a triangle by a factor of `v`."""
    if v < 1:
        raise ValueError(f"v {v} must be >= 1")
    int_weights = []
    for i in range(v + 1):
        for j in range(v + 1 - i):
            int_weights.append((i, j, v - (i + j)))
    int_weights = np.array(int_weights)
    weights = int_weights / v  # Barycentric weights.
    return weights


def tesselate_geodesic(base_verts, base_faces, v, eps=1e-4):
    """Tesselate the vertices of a geodesic polyhedron.
    Args:
      base_verts: tensor of floats, the vertex coordinates of the geodesic.
      base_faces: tensor of ints, the indices of the vertices of base_verts that
        constitute eachface of the polyhedra.
      v: int, the factor of the tesselation (v==1 is a no-op).
      eps: float, a small value used to determine if two vertices are the same.
    Returns:
      verts: a tensor of floats, the coordinates of the tesselated vertices.
    """
    if not isinstance(v, int):
        raise ValueError(f"v {v} must an integer")
    tri_weights = compute_tesselation_weights(v)

    verts = []
    for base_face in base_faces:
        new_verts = np.matmul(tri_weights, base_verts[base_face, :])
        new_verts /= np.sqrt(np.sum(new_verts**2, 1, keepdims=True))
        verts.append(new_verts)
    verts = np.concatenate(verts, 0)

    sq_dist = compute_sq_dist(verts.T)
    assignment = np.array([np.min(np.argwhere(d <= eps)) for d in sq_dist])
    unique = np.unique(assignment)
    verts = verts[unique, :]

    return verts


def generate_basis(base_shape, angular_tesselation, remove_symmetries=True, eps=1e-4):
    """Generates a 3D basis by tesselating a geometric polyhedron.
    Args:
      base_shape: string, the name of the starting polyhedron, must be either
        'icosahedron' or 'octahedron'.
      angular_tesselation: int, the number of times to tesselate the polyhedron,
        must be >= 1 (a value of 1 is a no-op to the polyhedron).
      remove_symmetries: bool, if True then remove the symmetric basis columns,
        which is usually a good idea because otherwise projections onto the basis
        will have redundant negative copies of each other.
      eps: float, a small number used to determine symmetries.
    Returns:
      basis: a matrix with shape [3, n].
    """
    if base_shape == "icosahedron":
        a = (np.sqrt(5) + 1) / 2
        verts = np.array(
            [
                (-1, 0, a),
                (1, 0, a),
                (-1, 0, -a),
                (1, 0, -a),
                (0, a, 1),
                (0, a, -1),
                (0, -a, 1),
                (0, -a, -1),
                (a, 1, 0),
                (-a, 1, 0),
                (a, -1, 0),
                (-a, -1, 0),
            ]
        ) / np.sqrt(a + 2)
        faces = np.array(
            [
                (0, 4, 1),
                (0, 9, 4),
                (9, 5, 4),
                (4, 5, 8),
                (4, 8, 1),
                (8, 10, 1),
                (8, 3, 10),
                (5, 3, 8),
                (5, 2, 3),
                (2, 7, 3),
                (7, 10, 3),
                (7, 6, 10),
                (7, 11, 6),
                (11, 0, 6),
                (0, 1, 6),
                (6, 1, 10),
                (9, 0, 11),
                (9, 11, 2),
                (9, 2, 5),
                (7, 2, 11),
            ]
        )
        verts = tesselate_geodesic(verts, faces, angular_tesselation)
    elif base_shape == "octahedron":
        verts = np.array(
            [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
        )
        corners = np.array(list(itertools.product([-1, 1], repeat=3)))
        pairs = np.argwhere(compute_sq_dist(corners.T, verts.T) == 2)
        faces = np.sort(np.reshape(pairs[:, 1], [3, -1]).T, 1)
        verts = tesselate_geodesic(verts, faces, angular_tesselation)
    else:
        raise ValueError(f"base_shape {base_shape} not supported")

    if remove_symmetries:
        # Remove elements of `verts` that are reflections of each other.
        match = compute_sq_dist(verts.T, -verts.T) < eps
        verts = verts[np.any(np.triu(match), 1), :]

    basis = verts[:, ::-1].copy()
    return torch.from_numpy(basis.T).to(dtype=torch.float32)

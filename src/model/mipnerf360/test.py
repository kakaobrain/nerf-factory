# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import functorch
import jax
import jax.numpy as jnp
import numpy as onp
import torch

f = lambda x: 3 * jnp.sin(x) + jnp.cos(x / 2)
f_torch = lambda x: 3 * torch.sin(x) + torch.cos(x / 2)

sampler = onp.arange(10)


def jax_linearize():
    ret = []
    input_tensor = 2.0
    test_tensor = jnp.array([3.0, 4.0, 5.0, 6.0, 7.0])
    y, f_jvp = jax.linearize(f, input_tensor)

    ret.append(y)
    for elem in test_tensor:
        ret.append(f_jvp(elem))

    return onp.array(ret)


def pytorch_linearize_nonvectorized():
    input_tensor = torch.tensor(2.0)
    test_tensor = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0])
    f_jvp = lambda x: torch.autograd.functional.jvp(f_torch, input_tensor, x)

    ret = []
    ret.append(f_jvp(input_tensor)[0].item())
    for elem in test_tensor:
        ret.append(f_jvp(elem)[-1].item())

    return onp.array(ret)


def pytorch_linearize_vectorized():

    input_tensor = torch.tensor([2.0], requires_grad=True)
    test_tensor = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0])

    ret = functorch.vjp(f_torch, input_tensor)[0]
    jacobian = functorch.vmap(functorch.jacrev(f_torch, argnums=0))(input_tensor)

    ret = torch.cat([ret, jacobian * torch.cat([test_tensor])])

    return ret.detach().numpy()


def contract_jnp(x):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    eps = jnp.finfo(jnp.float32).eps
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = jnp.maximum(eps, jnp.sum(x**2, axis=-1, keepdims=True))
    z = jnp.where(x_mag_sq <= 1, x, ((2 * jnp.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    return z


def track_linearize_jnp(mean, cov):
    """Apply function `fn` to a set of means and covariances, ala a Kalman filter.
    We can analytically transform a Gaussian parameterized by `mean` and `cov`
    with a function `fn` by linearizing `fn` around `mean`, and taking advantage
    of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
    https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).
    Args:
        fn: the function applied to the Gaussians parameterized by (mean, cov).
        mean: a tensor of means, where the last axis is the dimension.
        cov: a tensor of covariances, where the last two axes are the dimensions.
    Returns:
        fn_mean: the transformed means.
        fn_cov: the transformed covariances.
    """
    if (len(mean.shape) + 1) != len(cov.shape):
        raise ValueError("cov must be non-diagonal")
    fn_mean, lin_fn = jax.linearize(contract_jnp, mean)
    fn_cov = jax.vmap(lin_fn, -1, -2)(jax.vmap(lin_fn, -1, -2)(cov))
    return fn_mean, fn_cov


def random_mean_cov_generator():
    bsz, num_samples, dim = 32, 20, 3
    mean = onp.random.random((bsz, num_samples, dim)) * 1e10
    cov = onp.random.random((bsz, num_samples, dim, dim)) * 1e10
    return mean, cov


def contract(mean, cov):

    bsz, num_samples, dim = mean.shape

    def _contract(x):
        x_mag_sq = torch.sum(x**2, dim=-1, keepdim=True).clip(min=1e-32)
        z = torch.where(
            x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x
        )
        return z

    mean_reshape = mean.reshape(bsz * num_samples, dim)
    cov_reshape = cov.reshape(bsz * num_samples, dim, dim)
    ft_mean = functorch.vjp(_contract, mean)[0]
    ft_jacobian = functorch.vmap(functorch.jacrev(_contract, argnums=0))(mean_reshape)

    ft_cov = torch.einsum("bij, bjk -> bik", ft_jacobian, cov_reshape)
    ft_cov = torch.einsum("bij, bkj -> bik", ft_cov, ft_jacobian)

    return ft_mean.reshape(bsz, num_samples, dim), ft_cov.reshape(
        bsz, num_samples, dim, dim
    )


def lift_and_diagonalize(mean, cov, basis):
    """Project `mean` and `cov` onto basis and diagonalize the projected cov."""
    fn_mean = jnp.matmul(mean, basis, precision=jax.lax.Precision.HIGHEST)
    fn_cov_diag = jnp.sum(basis * jnp.matmul(cov, basis), axis=-2)
    return fn_mean, fn_cov_diag


# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for constructing geodesic polyhedron, which are used as a basis."""

import itertools

import numpy as np


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

    basis = verts[:, ::-1]
    return basis.T


if __name__ == "__main__":

    ret_jax = jax_linearize()
    ret_torch_novec = pytorch_linearize_nonvectorized()
    ret_torch_vec = pytorch_linearize_vectorized()

    onp.testing.assert_almost_equal(ret_jax, ret_torch_novec, decimal=4)
    onp.testing.assert_almost_equal(ret_torch_novec, ret_torch_vec, decimal=4)
    onp.testing.assert_almost_equal(ret_torch_novec, ret_jax, decimal=4)

    test_trial = 10

    for i in range(test_trial):
        mean, cov = random_mean_cov_generator()
        fn_mean_jnp, fn_cov_jnp = track_linearize_jnp(jnp.array(mean), jnp.array(cov))
        fn_mean_torch, fn_cov_torch = contract(
            torch.from_numpy(mean), torch.from_numpy(cov)
        )
        fn_mean_jnp_onp, fn_cov_jnp_onp = onp.array(fn_mean_jnp), onp.array(fn_cov_jnp)
        fn_mean_torch_onp, fn_cov_torch_onp = (
            fn_mean_torch.detach().numpy(),
            fn_cov_torch.detach().numpy(),
        )
        onp.testing.assert_almost_equal(fn_mean_torch_onp, fn_mean_jnp_onp, decimal=3)
        onp.testing.assert_almost_equal(fn_cov_torch_onp, fn_cov_jnp_onp, decimal=3)

    basis = generate_basis("icosahedron", 2)
    lift_and_diagonalize(fn_mean_jnp, fn_cov_jnp, basis)

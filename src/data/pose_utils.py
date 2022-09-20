import numpy as np


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses: np.ndarray):
    """Transforms poses so principal components lie on XYZ axes.
    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / (np.linalg.norm(x) + 1e-7)


def generate_ellipse_path(
    poses: np.ndarray,
    n_frames: int = 5,
    const_speed: bool = True,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                ),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def pose_interp(poses, factor):

    pose_list = []
    for i in range(len(poses)):
        pose_list.append(poses[i])

        if i == len(poses) - 1:
            factor = 4 * factor

        next_idx = (i + 1) % len(poses)
        axis, angle = R_to_axis_angle(
            (poses[next_idx, :3, :3] @ poses[i, :3, :3].T)[None]
        )
        for j in range(factor - 1):
            ret = np.eye(4)
            j_fact = (j + 1) / factor
            angle_j = angle * j_fact
            pose_rot = R_axis_angle(angle_j, axis)
            ret[:3, :3] = pose_rot @ poses[i, :3, :3]
            trans_t = (1 - j_fact) * poses[i, :3, 3] + (j_fact) * poses[next_idx, :3, 3]
            ret[:3, 3] = trans_t
            pose_list.append(ret)

    return np.stack(pose_list)


def R_axis_angle(angle, axis):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """
    len_angle = len(angle)
    matrix = np.zeros((len_angle, 3, 3))

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    # Multiplications (to remove duplicate calculations).
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # Update the rotation matrix.
    matrix[:, 0, 0] = x * xC + ca
    matrix[:, 0, 1] = xyC - zs
    matrix[:, 0, 2] = zxC + ys
    matrix[:, 1, 0] = xyC + zs
    matrix[:, 1, 1] = y * yC + ca
    matrix[:, 1, 2] = yzC - xs
    matrix[:, 2, 0] = zxC - ys
    matrix[:, 2, 1] = yzC + xs
    matrix[:, 2, 2] = z * zC + ca

    return matrix


def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    len_matrix = len(matrix)
    axis = np.zeros((len_matrix, 3))
    axis[:, 0] = matrix[:, 2, 1] - matrix[:, 1, 2]
    axis[:, 1] = matrix[:, 0, 2] - matrix[:, 2, 0]
    axis[:, 2] = matrix[:, 1, 0] - matrix[:, 0, 1]

    # Angle.
    r = np.hypot(axis[:, 0], np.hypot(axis[:, 1], axis[:, 2]))
    t = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    theta = np.arctan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r[:, None]

    # Return the data.
    return axis, theta

# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF++ (https://github.com/Kai-46/nerfplusplus)
# Copyright (c) 2020 the NeRF++ authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

import glob
import os
from typing import *

import imageio
import numpy as np


def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def similarity_from_cameras(c2w):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale


def load_lf_data(
    datadir: str,
    scene_name: str,
    train_skip: int,
    val_skip: int,
    test_skip: int,
    cam_scale_factor: float,
    near: Optional[float],
    far: Optional[float],
):

    basedir = os.path.join(datadir, scene_name)

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    # camera parameters files
    intrinsics_files = find_files(
        "{}/train/intrinsics".format(basedir), exts=["*.txt"]
    )[::train_skip]
    intrinsics_files += find_files(
        "{}/validation/intrinsics".format(basedir), exts=["*.txt"]
    )[::val_skip]
    intrinsics_files += find_files(
        "{}/test/intrinsics".format(basedir), exts=["*.txt"]
    )[::test_skip]
    pose_files = find_files("{}/train/pose".format(basedir), exts=["*.txt"])[
        ::train_skip
    ]
    pose_files += find_files("{}/validation/pose".format(basedir), exts=["*.txt"])[
        ::val_skip
    ]
    pose_files += find_files("{}/test/pose".format(basedir), exts=["*.txt"])[
        ::test_skip
    ]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files("{}/rgb".format(basedir), exts=["*.png", "*.jpg"])
    if len(img_files) > 0:
        assert len(img_files) == cam_cnt
    else:
        img_files = [
            None,
        ] * cam_cnt

    # assume all images have the same size as training image
    train_imgfile = find_files("{}/train/rgb".format(basedir), exts=["*.png", "*.jpg"])[
        ::train_skip
    ]
    val_imgfile = find_files(
        "{}/validation/rgb".format(basedir), exts=["*.png", "*.jpg"]
    )[::val_skip]
    test_imgfile = find_files("{}/test/rgb".format(basedir), exts=["*.png", "*.jpg"])[
        ::test_skip
    ]
    i_train = np.arange(len(train_imgfile))
    i_val = np.arange(len(val_imgfile)) + len(train_imgfile)
    i_test = np.arange(len(test_imgfile)) + len(train_imgfile) + len(val_imgfile)
    i_all = np.arange(len(train_imgfile) + len(val_imgfile) + len(test_imgfile))
    i_split = (i_train, i_val, i_test, i_all)

    images = (
        np.stack(
            [
                imageio.imread(imgfile)
                for imgfile in train_imgfile + val_imgfile + test_imgfile
            ]
        )
        / 255.0
    )
    h, w = images[0].shape[:2]

    intrinsics = np.stack(
        [parse_txt(intrinsics_file) for intrinsics_file in intrinsics_files]
    )
    extrinsics = np.stack([parse_txt(pose_file) for pose_file in pose_files])

    if cam_scale_factor > 0:
        T, sscale = similarity_from_cameras(extrinsics)
        extrinsics = np.einsum("nij, ki -> nkj", extrinsics, T)
        scene_scale = cam_scale_factor * sscale
        extrinsics[:, :3, 3] *= scene_scale

    num_frame = len(extrinsics)

    image_sizes = np.array([[h, w] for i in range(num_frame)])

    near = 0.0 if near is None else near
    far = 1.0 if far is None else far

    render_poses = extrinsics

    return (
        images,
        intrinsics,
        extrinsics,
        image_sizes,
        near,
        far,
        (-1, -1),
        i_split,
        render_poses,
    )

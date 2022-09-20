# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Ref-NeRF (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import json
import os

import gdown
import imageio
import numpy as np
import torch

trans_t = lambda t: torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).float()
        @ c2w
    )
    return c2w


def load_shiny_blender_data(
    datadir: str,
    scene_name: str,
    train_skip: int,
    val_skip: int,
    test_skip: int,
    cam_scale_factor: float,
    white_bkgd: bool,
):
    basedir = os.path.join(datadir, scene_name)
    cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        if s == "val":
            continue
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)
    metas["val"] = metas["test"]

    images = []
    normals = []
    extrinsics = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        norms = []
        alphas = []
        poses = []

        if s == "train":
            skip = train_skip
        elif s == "val":
            skip = val_skip
        elif s == "test":
            skip = test_skip

        for frame in meta["frames"][::skip]:
            img_fname = os.path.join(basedir, frame["file_path"] + ".png")
            norm_fname = os.path.join(basedir, frame["file_path"] + "_normal.png")
            imgs.append(imageio.imread(img_fname))
            norms.append(imageio.imread(norm_fname))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        alphas = imgs[..., -1:]
        norms = np.array(norms).astype(np.float32)[..., :3] * 2.0 / 255.0 - 1.0
        # Concatenate normals and alphas for computing MAE
        norms = np.concatenate([norms, alphas], axis=-1)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        images.append(imgs)
        normals.append(norms)
        extrinsics.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    images = np.concatenate(images, 0)
    normals = np.concatenate(normals, 0)

    extrinsics = np.concatenate(extrinsics, 0)

    extrinsics[:, :3, 3] *= cam_scale_factor
    extrinsics = extrinsics @ cam_trans

    h, w = imgs[0].shape[:2]
    num_frame = len(extrinsics)
    i_split += [np.arange(num_frame)]

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    intrinsics = np.array(
        [
            [[focal, 0.0, 0.5 * w], [0.0, focal, 0.5 * h], [0.0, 0.0, 1.0]]
            for _ in range(num_frame)
        ]
    )
    image_sizes = np.array([[h, w] for _ in range(num_frame)])

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0) @ cam_trans
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )
    render_poses[:, :3, 3] *= cam_scale_factor
    near = 2.0
    far = 6.0

    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

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
        normals,
    )

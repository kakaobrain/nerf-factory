# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import json
import os

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


def load_blender_ms_data(
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

    metadatapath = os.path.join(basedir, "metadata.json")
    with open(metadatapath) as fp:
        metadata = json.load(fp)

    images = []
    extrinsics = []
    counts = [0]
    focals = []
    multlosses = []

    for s in splits:
        meta = metadata[s]
        imgs = []
        poses = []
        fs = []
        multloss = []

        if s == "train":
            skip = train_skip
        elif s == "val":
            skip = val_skip
        elif s == "test":
            skip = test_skip

        for (filepath, pose, focal, mult) in zip(
            meta["file_path"][::skip],
            meta["cam2world"][::skip],
            meta["focal"][::skip],
            meta["lossmult"][::skip],
        ):
            fname = os.path.join(basedir, filepath)
            imgs.append(imageio.imread(fname))
            poses.append(np.array(pose))
            fs.append(focal)
            multloss.append(mult)

        imgs = [(img / 255.0).astype(np.float32) for img in imgs]
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + len(imgs))
        images += imgs
        focals += fs
        extrinsics.append(poses)
        multlosses.append(np.array(multloss))

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    extrinsics = np.concatenate(extrinsics, 0)

    extrinsics[:, :3, 3] *= cam_scale_factor
    extrinsics = extrinsics @ cam_trans

    image_sizes = np.array([img.shape[:2] for img in images])
    num_frame = len(extrinsics)
    i_split += [np.arange(num_frame)]

    intrinsics = np.array(
        [
            [[focal, 0.0, 0.5 * w], [0.0, focal, 0.5 * h], [0.0, 0.0, 1.0]]
            for (focal, (h, w)) in zip(focals, image_sizes)
        ]
    )

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
        images = [
            image[..., :3] * image[..., -1:] + (1.0 - image[..., -1:])
            for image in images
        ]
    else:
        images = [image[..., :3] for image in images]

    multlosses = np.concatenate(multlosses)

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
        multlosses,  # Train only
    )

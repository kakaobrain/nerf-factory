# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import gin
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import SequentialSampler


class DDPSampler(SequentialSampler):
    def __init__(self, batch_size, num_replicas, rank, tpu):
        self.data_source = None
        self.batch_size = batch_size
        self.drop_last = False
        ngpus = torch.cuda.device_count()
        if ngpus == 1 and not tpu:
            rank, num_replicas = 0, 1
        else:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
        self.rank = rank
        self.num_replicas = num_replicas


class DDPSequnetialSampler(DDPSampler):
    def __init__(self, batch_size, num_replicas, rank, N_total, tpu):
        self.N_total = N_total
        super(DDPSequnetialSampler, self).__init__(batch_size, num_replicas, rank, tpu)

    def __iter__(self):
        idx_list = np.arange(self.N_total)
        return iter(idx_list[self.rank :: self.num_replicas])

    def __len__(self):
        return int(np.ceil(self.N_total / self.num_replicas))


class SingleImageDDPSampler(DDPSampler):
    def __init__(
        self,
        batch_size,
        num_replicas,
        rank,
        N_img,
        N_pixels,
        epoch_size,
        tpu,
        precrop,
        precrop_steps,
    ):
        super(SingleImageDDPSampler, self).__init__(batch_size, num_replicas, rank, tpu)
        self.N_pixels = N_pixels
        self.N_img = N_img
        self.epoch_size = epoch_size
        self.precrop = precrop
        self.precrop_steps = precrop_steps

    def __iter__(self):
        image_choice = np.random.choice(
            np.arange(self.N_img), self.epoch_size, replace=True
        )
        image_shape = self.N_pixels[image_choice]
        if not self.precrop:
            idx_choice = [
                np.random.choice(
                    np.arange(image_shape[i, 0] * image_shape[i, 1]), self.batch_size
                )
                for i in range(self.epoch_size)
            ]
        else:
            idx_choice = []
            h_pick = [
                np.random.choice(np.arange(image_shape[i, 0] // 2), self.batch_size)
                + image_shape[i, 0] // 4
                for i in range(self.precrop_steps)
            ]
            w_pick = [
                np.random.choice(np.arange(image_shape[i, 1] // 2), self.batch_size)
                + image_shape[i, 1] // 4
                for i in range(self.precrop_steps)
            ]
            idx_choice = [
                h_pick[i] * image_shape[i, 1] + w_pick[i]
                for i in range(self.precrop_steps)
            ]

            idx_choice += [
                np.random.choice(
                    np.arange(image_shape[i, 0] * image_shape[i, 1]), self.batch_size
                )
                for i in range(self.epoch_size - self.precrop_steps)
            ]
            self.precrop = False

        idx_choice = np.stack(idx_choice)
        idx_jump = np.concatenate(
            [
                np.zeros_like(self.N_pixels[0]),
                np.cumsum(self.N_pixels[:-1, 0] * self.N_pixels[:-1, 1]),
            ]
        )[..., None]
        idx_shift = idx_jump[image_choice] + idx_choice
        idx_shift = idx_shift[:, self.rank :: self.num_replicas]

        return iter(idx_shift)

    def __len__(self):
        return self.epoch_size


class MultipleImageDDPSampler(DDPSampler):
    def __init__(self, batch_size, num_replicas, rank, total_len, epoch_size, tpu):
        super(MultipleImageDDPSampler, self).__init__(
            batch_size, num_replicas, rank, tpu
        )
        self.total_len = total_len
        self.epoch_size = epoch_size

    def __iter__(self):
        full_index = np.arange(self.total_len)
        indices = np.stack(
            [
                np.random.choice(full_index, self.batch_size)
                for _ in range(self.epoch_size)
            ]
        )
        for batch in indices:
            yield batch[self.rank :: self.num_replicas]

    def __len__(self):
        return self.epoch_size


@gin.configurable()
class MultipleImageDynamicDDPSampler(DDPSampler):
    def __init__(
        self,
        batch_size,
        num_replicas,
        rank,
        total_len,
        N_img,
        N_pixels,
        epoch_size,
        tpu,
        N_coarse=0,
    ):
        super(MultipleImageDynamicDDPSampler, self).__init__(
            batch_size, num_replicas, rank, tpu
        )
        self.total_len = total_len
        self.epoch_size = epoch_size
        self.N_pixels = N_pixels
        self.N_img = N_img
        self.N_coarse = N_coarse

    def __iter__(self):
        indices = np.random.rand(self.epoch_size - self.N_coarse, self.batch_size)

        image_choice = np.random.choice(
            np.arange(self.N_img), self.N_coarse, replace=True
        )
        image_shape = self.N_pixels[image_choice]
        idx_choice = [
            np.random.choice(
                np.arange(image_shape[i, 0] * image_shape[i, 1]), self.batch_size
            )
            for i in range(self.N_coarse)
        ]

        idx_choice = np.stack(idx_choice)
        idx_jump = np.concatenate(
            [
                np.zeros_like(self.N_pixels[0]),
                np.cumsum(self.N_pixels[:-1, 0] * self.N_pixels[:-1, 1]),
            ]
        )[..., None]
        idx_shift = idx_jump[image_choice] + idx_choice
        idx_shift = idx_shift[:, self.rank :: self.num_replicas]
        for batch in idx_shift:
            yield batch

        for batch in indices:
            yield np.floor(
                (batch[self.rank :: self.num_replicas]) * self.total_len
            ).astype(np.uint)

    def __len__(self):
        return self.epoch_size

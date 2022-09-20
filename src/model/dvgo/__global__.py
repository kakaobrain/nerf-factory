# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from DVGO (https://github.com/sunset1995/DirectVoxGO)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os

from torch.utils.cpp_extension import load

root_dir = __file__.split(os.path.relpath(__file__))[0]

render_utils_cuda = None
total_variation_cuda = None
ub360_utils_cuda = None
adam_upd_cuda = None

sources = ["lib/dvgo/cuda/adam_upd.cpp", "lib/dvgo/cuda/adam_upd_kernel.cu"]


def init():
    global render_utils_cuda
    render_utils_cuda = load(
        name="render_utils_cuda",
        sources=[
            os.path.join(root_dir, path)
            for path in [
                "lib/dvgo/cuda/render_utils.cpp",
                "lib/dvgo/cuda/render_utils_kernel.cu",
            ]
        ],
        verbose=True,
    )

    global ub360_utils_cuda
    ub360_utils_cuda = load(
        name="ub360_utils_cuda",
        sources=[
            os.path.join(root_dir, path)
            for path in [
                "lib/dvgo/cuda/ub360_utils.cpp",
                "lib/dvgo/cuda/ub360_utils_kernel.cu",
            ]
        ],
        verbose=True,
    )

    global total_variation_cuda
    if total_variation_cuda is None:
        total_variation_cuda = load(
            name="total_variation_cuda",
            sources=[
                os.path.join(root_dir, path)
                for path in [
                    "lib/dvgo/cuda/total_variation.cpp",
                    "lib/dvgo/cuda/total_variation_kernel.cu",
                ]
            ],
            verbose=True,
        )

    global adam_upd_cuda
    adam_upd_cuda = load(
        name="adam_upd_cuda",
        sources=[os.path.join(root_dir, path) for path in sources],
        verbose=True,
    )

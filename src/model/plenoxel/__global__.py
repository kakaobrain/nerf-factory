# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Plenoxels (https://github.com/sxyu/svox2)
# Copyright (c) 2022 the Plenoxel authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

BASIS_TYPE_SH = 1
BASIS_TYPE_3D_TEXTURE = 4
BASIS_TYPE_MLP = 255


def _get_c_extension():
    from warnings import warn

    try:
        import lib.plenoxel as _C

        if not hasattr(_C, "sample_grid"):
            _C = None
    except:
        _C = None

    return _C

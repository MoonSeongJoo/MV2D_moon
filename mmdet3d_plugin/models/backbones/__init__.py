# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .vovnetcp import VoVNetCP
from .voxelnet import SimpleVoxelization, SimpleVoxelNet 

__all__ = ['SimpleVoxelization','SimpleVoxelNet']
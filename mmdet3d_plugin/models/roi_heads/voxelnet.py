import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import Voxelization
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS

from mmdet3d_plugin.models.voxel_encoders.voxel_encoder import HardSimpleVFE
from mmdet3d_plugin.models.middle_encoders.pillar_scatter import PointPillarsScatter
from mmdet3d_plugin.models.backbones.second import SECOND

@HEADS.register_module()
class SimpleVoxelization(nn.Module):
    def __init__(self, 
                 voxel_size=[0.2, 0.2, 8], 
                 point_cloud_range=[0, -40, -3, 70.4, 40, 1], 
                 max_num_points=32, max_voxels=(16000, 40000)):
        
        super().__init__()
        self.voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        )
    
    @torch.no_grad()
    def forward(self, points):
        # points는 [batch_size, N, ndim] list/배치
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # 배치 차원 부여
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels,coors_batch, num_points

@HEADS.register_module()
class SimpleVoxelNet(BaseModule):
    def __init__(self,init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # Voxel feature encoder
        self.voxel_encoder = HardSimpleVFE(num_features=4)
        # Middle encoder (BEV 변환)
        self.middle_encoder = PointPillarsScatter(in_channels=4, output_shape=[900, 1600])
       
        # 2D BEV backbone
        self.pts_backbone = SECOND(
            in_channels=4,
            layer_nums=[3, 5],
            layer_strides=[2, 2],
            out_channels=[64, 128],
        )

    def forward(self, voxels, coors, num_points):
        if not torch.is_tensor(num_points):
            coors = torch.tensor(coors, device=voxels.device)
            num_points = torch.tensor(num_points, device=voxels.device)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)  # [num_voxels, C]
        x = self.middle_encoder(voxel_features, coors, batch_size=1)
        x = self.pts_backbone(x)

        # 필요 없다면 반환 직후 caller에서 반드시 아래처럼 관리!
        del voxel_features, num_points, coors, voxels
        torch.cuda.empty_cache()
        
        return x  # [B, C, H, W]

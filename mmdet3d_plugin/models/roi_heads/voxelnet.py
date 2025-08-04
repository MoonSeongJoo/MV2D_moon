import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import Voxelization 

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
class SimpleVoxelNet(nn.Module):
    def __init__(self,load_pretrained_path=None, device='cpu'):
        super().__init__()
        
        # Voxel feature encoder
        self.voxel_encoder = HardSimpleVFE(num_features=4)
        # Middle encoder (BEV 변환)
        self.middle_encoder = PointPillarsScatter(in_channels=4, output_shape=[900, 1600])
        # 2D BEV backbone
        self.backbone_3d = SECOND(
            in_channels=4,
            layer_nums=[3, 5],
            layer_strides=[2, 2],
            out_channels=[64, 128]
        )
                # pretrained 가중치 로드
        if load_pretrained_path is not None:
            self._load_pretrained_weights(load_pretrained_path, device)

    def _load_pretrained_weights(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            pretrained_sd = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_sd = checkpoint['state_dict']
        else:
            pretrained_sd = checkpoint

        model_sd = self.state_dict()
        new_sd = {}
        required_prefix = 'roi_head.lidar_voxelnet.'

        for k, v in pretrained_sd.items():
            # 'model_state', 'global_step' 등 메타키 건너뛰기
            if not isinstance(k, str) or '.' not in k:
                continue
            new_k = required_prefix + k if not k.startswith(required_prefix) else k
            if new_k in model_sd and model_sd[new_k].shape == v.shape:
                new_sd[new_k] = v

        load_res = self.load_state_dict(new_sd, strict=False)
        print(f"Pretrained weights loaded with missing keys: {load_res.missing_keys}")
        print(f"Pretrained weights loaded with unexpected keys: {load_res.unexpected_keys}")
        print("loaded end")


    def forward(self, voxels, coors, num_points):
        if not torch.is_tensor(num_points):
            coors = torch.tensor(coors, device=voxels.device)
            num_points = torch.tensor(num_points, device=voxels.device)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)  # [num_voxels, C]
        x = self.middle_encoder(voxel_features, coors, batch_size=1)
        x = self.backbone_3d(x)
        return x  # [B, C, H, W]

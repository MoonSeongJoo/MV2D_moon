import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import Voxelization 

from mmdet.models.builder import BACKBONELIDAR

from mmdet3d_plugin.models.voxel_encoders.voxel_encoder import HardSimpleVFE
from mmdet3d_plugin.models.middle_encoders.pillar_scatter import PointPillarsScatter
from mmdet3d.models.voxel_encoders import PillarFeatureNet
from mmdet3d_plugin.models.backbones.second import SECOND

@BACKBONELIDAR.register_module()
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
    
    # @torch.no_grad()
    # def forward(self, points):
    #     # points는 [batch_size, N, ndim] list/배치
    #     voxels, coors, num_points = [], [], []
    #     for res in points:
    #         res_voxels, res_coors, res_num_points = self.voxel_layer(res)
    #         voxels.append(res_voxels)
    #         coors.append(res_coors)
    #         num_points.append(res_num_points)
    #     voxels = torch.cat(voxels, dim=0)
    #     num_points = torch.cat(num_points, dim=0)
    #     coors_batch = []
    #     for i, coor in enumerate(coors):
    #         # 배치 차원 부여
    #         coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
    #         coors_batch.append(coor_pad)
    #     coors_batch = torch.cat(coors_batch, dim=0)
    #     return voxels,coors_batch, num_points
    
    @torch.no_grad()
    def forward(self, points):
        # points: [batch_size, N, ndim] list/배치
        voxels_list = []
        coors_list = []
        num_points_list = []

        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels_list.append(res_voxels)
            coors_list.append(res_coors)
            num_points_list.append(res_num_points)

        # 리스트가 모두 텐서이므로 효율적으로 cat
        voxels = torch.cat(voxels_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)

        # coors_batch 구축: 배치 인덱스 부여
        coors_batch_list = []
        for i, coor in enumerate(coors_list):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch_list.append(coor_pad)
        coors_batch = torch.cat(coors_batch_list, dim=0)

        # 모든 리스트는 반환 직전 del로 참조 해제 → 메모리 누수 최소화
        del voxels_list, coors_list, num_points_list, coors_batch_list

        # (필요시: detach/clone으로 gradient 및 참조 끊기)
        # voxels = voxels.detach().clone()
        # num_points = num_points.detach().clone()
        # coors_batch = coors_batch.detach().clone()

        return voxels, coors_batch, num_points


@BACKBONELIDAR.register_module()
class SimpleVoxelNet(nn.Module):
    def __init__(self,load_pretrained_path=None, device='cpu'):
        super().__init__()
        # Voxel feature encoder
        # self.voxel_encoder = HardSimpleVFE(num_features=4)
        self.voxel_encoder = PillarFeatureNet(
            in_channels=4,          # ✅ 수정: 'num_in_features' -> 'in_channels'
            feat_channels=(64,),    # ✅ 수정: 'feat_chns' -> 'feat_channels', 이 인자가 출력 채널을 64로 정의합니다.
            with_distance=False,
            voxel_size=(0.2, 0.2, 8),
            point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
            # with_cluster_center, with_voxel_center 등 다른 인자들은 기본값(True)이 사용됩니다.
        )
        # Middle encoder (BEV 변환)
        self.middle_encoder = PointPillarsScatter(in_channels=64, output_shape=[900, 1600])
        # 2D BEV backbone
        # self.backbone_3d = SECOND(
        #     in_channels=4,
        #     layer_nums=[3, 5],
        #     layer_strides=[2, 2],
        #     out_channels=[64, 128]
        # )
        self.backbone_3d=SECOND(
            in_channels=64, # 이 부분을 64로 설정
            out_channels=[64, 128, 256], # 이 부분도 pre-trained 모델과 동일하게 설정
            layer_nums=[3, 5, 5],
            layer_strides=[1, 2, 2] # stride 값도 확인 필요
        )

        # pretrained 가중치 로드
        if load_pretrained_path is not None:
            self._load_pretrained_weights(load_pretrained_path, device)

    def _load_pretrained_weights(self, checkpoint_path, device):
        """
        미리 키가 변환된 state_dict 파일을 모델에 로드합니다.
        """
        print(f"✅ Loading pre-processed weights from: {checkpoint_path}")

        # 1. 이미 처리된 가중치 파일(.pth)을 불러옵니다.
        #    이전 스크립트에서 state_dict 자체를 저장했으므로, 바로 딕셔너리가 로드됩니다.
        pretrained_sd = torch.load(checkpoint_path, map_location=device)

        # 2. 모델에 state_dict를 직접 로드합니다.
        #    strict=False 옵션은 일부 키가 없거나 추가로 있어도 오류 없이 로드를 진행시킵니다.
        load_result = self.load_state_dict(pretrained_sd, strict=False)

        # 3. 로드 결과를 상세히 출력하여 확인합니다.
        print("--- Pre-trained weights load result ---")
        if load_result.missing_keys:
            print(f"⚠️ Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"⚠️ Unexpected keys: {load_result.unexpected_keys}")

        if not load_result.missing_keys and not load_result.unexpected_keys:
            print("🎉 All keys matched and loaded successfully!")
        else:
            print("Partial load complete.")
        
        print("---------------------------------------")

    # def forward(self, voxels, coors, num_points):
    #     if not torch.is_tensor(num_points):
    #         coors = torch.tensor(coors, device=voxels.device)
    #         num_points = torch.tensor(num_points, device=voxels.device)
    #     voxel_features = self.voxel_encoder(voxels, num_points, coors)  # [num_voxels, C]
    #     x = self.middle_encoder(voxel_features, coors, batch_size=1)
    #     x = self.backbone_3d(x)
    #     return x  # [B, C, H, W]
    
    def forward(self, voxels, coors, num_points):
        # num_points, coors가 이미 torch.Tensor이면 재생성하지 않음
        if not torch.is_tensor(num_points):
            num_points = torch.as_tensor(num_points, device=voxels.device)
        if not torch.is_tensor(coors):
            coors = torch.as_tensor(coors, device=voxels.device)

        # Voxel feature extraction
        voxel_features = self.voxel_encoder(voxels, num_points, coors)  # [num_voxels, C]
        # Middle encoder
        x = self.middle_encoder(voxel_features, coors, batch_size=1)
        # Backbone
        x = self.backbone_3d(x)

        # 필요 없다면 반환 직후 caller에서 반드시 아래처럼 관리!
        del voxel_features, num_points, coors, voxels
        torch.cuda.empty_cache()

        return x  # [B, C, H, W]

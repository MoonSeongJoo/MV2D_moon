import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from mmcv.runner import BaseModule

@HEADS.register_module()
class ZEstimator(BaseModule):
    def __init__(self, enc_channels=312, bbox_channels=256, uv_dim=2, hidden_dim=512,init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # enc_out 특징 압축
        self.enc_adaptor = nn.Sequential(
            nn.Conv2d(enc_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # bbox_feats 처리
        self.bbox_adaptor = nn.Sequential(
            nn.Conv2d(bbox_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # UV 좌표 임베딩
        self.uv_embed = nn.Linear(uv_dim, 64)
        
        # 최종 융합 및 깊이 예측
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, uv, depth_map, bbox_feats, enc_out):
        """
        Args:
            uv: [N, 4] (cam_id, obj_id, u, v)
            depth_map: [num_cams, H, W] 
            bbox_feats: [N, 256, 7, 7]
            enc_out: [B, C, H, W] = [6, 312, 12, 64]
        """
        N = uv.size(0)
        
        # 1. cam_ids 추출 (uv의 첫 번째 열)
        cam_ids = uv[:, 0].long()  # [N]
        
        # 2. enc_out 특징 처리
        enc_reduced = self.enc_adaptor(enc_out)  # [6, 128, 1, 1]
        enc_reduced = enc_reduced.squeeze(-1).squeeze(-1)  # [6, 128]
        
        # 객체별 enc 특징 선택
        object_enc = enc_reduced[cam_ids]  # [N, 128]
        
        # 3. bbox_feats 처리
        bbox_reduced = self.bbox_adaptor(bbox_feats)  # [N, 128, 1, 1]
        bbox_reduced = bbox_reduced.squeeze(-1).squeeze(-1)  # [N, 128]
        
        # 4. UV 좌표 처리 (uv의 3-4열 사용)
        uv_coords = uv[:, 2:4]  # [N, 2]
        uv_embedded = self.uv_embed(uv_coords)  # [N, 64]
        
        # 5. 특징 융합
        combined = torch.cat([object_enc, bbox_reduced, uv_embedded], dim=1)
        fused = self.fusion(combined)
        
        # 6. 깊이 예측
        z_estimated_real = self.depth_predictor(fused).squeeze(1)
        # z_estimated_real = self.denormalize_depth(z_estimated_normalized)
        
        # 7. 실제 LiDAR 깊이 조회
        H, W = depth_map.shape[1], depth_map.shape[2]
        u = uv[:, 2].clamp(0, W-1).long()
        v = uv[:, 3].clamp(0, H-1).long()
        z_depth_real = depth_map[cam_ids, v, u]
        
        # 8. 신뢰도 기반 융합
        lidar_confidence = self.estimate_lidar_confidence(z_depth_real)
        valid_lidar_mask = (z_depth_real > 0)
        lidar_confidence_adjusted = lidar_confidence * valid_lidar_mask.float()
        
        z_final_real = (
            lidar_confidence_adjusted * z_depth_real +
            (1 - lidar_confidence_adjusted) * z_estimated_real
        )
        
        return {
            'depth': z_final_real.unsqueeze(-1),
            'z_lidar_real': z_depth_real,
            'confidence': lidar_confidence,
            'z_estimated_real': z_estimated_real
        }

    def denormalize_depth(self, normalized_depth):
        """정규화된 깊이를 실제 스케일로 변환"""
        # 실제 구현시 데이터셋 통계 기반 조정
        return normalized_depth * 80.0  # 예시: 0~1 → 0~80m

    def estimate_lidar_confidence(self, z_depth_real):
        """LiDAR 신뢰도 추정 (깊이 기반)"""
        # 실제 구현시 깊이, 강도 등 활용
        confidence = torch.sigmoid(0.1 * (50 - z_depth_real))
        return confidence

class BBoxEnhancedZEstimator(nn.Module):
    def __init__(self, depth_shape, bbox_feat_dim=256*7*7, hidden_dim=128):
        super().__init__()
        self.max_u = depth_shape[1]  # 이미지 너비 (W)
        self.max_v = depth_shape[0]  # 이미지 높이 (H)
        
        # 1. UV 좌표 임베딩 레이어
        self.uv_embed = nn.Embedding(self.max_u * self.max_v, 1)
        
        # 2. BBox 특징 MLP
        self.bbox_mlp = nn.Sequential(
            nn.Linear(bbox_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.xavier_uniform_(self.bbox_mlp[0].weight)
        nn.init.constant_(self.bbox_mlp[2].bias, 0.0)
        
        # 3. 동적 가중치 파라미터
        self.alpha = nn.Parameter(torch.tensor([0.7]))

    def forward(self, uv, depth_map, bbox_feats):
        """
        Input Shapes:
        - uv: [N, 4] (cam_id, obj_id, u, v)
        - depth_map: [num_cams, H, W] 
        - bbox_feats: [N, 256, 7, 7]
        Returns: [N, 1]
        """
        N = uv.size(0)
        H, W = depth_map.shape[1], depth_map.shape[2]  # [num_cams, H, W]

        # 1. 입력 분해
        cam_ids = uv[:, 0].long()  # [N]
        u = uv[:, 2].clamp(0, W-1).long()  # [N]
        v = uv[:, 3].clamp(0, H-1).long()  # [N]

        # 2. 해당 카메라의 depth 값 조회
        z_depth = depth_map[cam_ids, v, u]  # [N]

        # 3. UV 임베딩 조회
        indices = u * H + v  # [N]
        z_uv = self.uv_embed(indices).squeeze(1)  # [N]

        # 4. BBox 특징 처리
        bbox_flat = bbox_feats.view(N, -1)  # [N, 256*7*7]
        z_bbox = self.bbox_mlp(bbox_flat).squeeze(1)  # [N]

        # 5. 최종 예측
        valid_mask = z_depth > 0
        z_final = torch.where(
            valid_mask,
            z_depth,
            self.alpha * z_uv + (1 - self.alpha) * z_bbox
        )

        return z_final.unsqueeze(-1)  # [N, 1]
    
# class ConfidenceEstimator(nn.Module):
#     """라이다 실측치의 신뢰도를 추정하는 모듈"""
#     def __init__(self, input_dim=256):
#         super().__init__()
#         self.confidence_net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, 1),
#             nn.Sigmoid()  # 0~1 범위의 confidence
#         )
        
#     def forward(self, bbox_feats, depth_context):
#         """
#         Args:
#             bbox_feats: [N, 256, 7, 7] bbox features
#             depth_context: [N, context_dim] depth 관련 context features
#         Returns:
#             confidence: [N, 1] confidence scores
#         """
#         bbox_flat = bbox_feats.view(bbox_feats.size(0), -1)
#         combined_feats = torch.cat([bbox_flat, depth_context], dim=1)
#         confidence = self.confidence_net(combined_feats)
#         return confidence

class DynamicWeightingNetwork(nn.Module):
    """동적 가중치를 학습하는 네트워크"""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0~1 범위의 alpha 값
        )
        
    def forward(self, uv_feats, bbox_feats):
        """
        Args:
            uv_feats: [N, feature_dim] UV embedding features
            bbox_feats: [N, feature_dim] BBox features  
        Returns:
            alpha: [N, 1] dynamic weighting factor
        """
        combined = torch.cat([uv_feats, bbox_feats], dim=1)
        alpha = self.weight_net(combined)
        return alpha

class ImprovedDepthEstimator(nn.Module):
    def __init__(self, max_image_size=(2048, 2048), bbox_feat_dim=256*7*7, 
                 hidden_dim=512, confidence_threshold=0.3,
                 depth_scale_factor=80.0, depth_shift=0.1):  # 스케일 파라미터 추가
        super().__init__()
        
        # 기존 코드...
        self.max_h, self.max_w = max_image_size
        self.hash_embed_size = 50000
        self.uv_embed = nn.Embedding(self.hash_embed_size, 256)
        
        # Depth normalization parameters
        self.depth_scale_factor = depth_scale_factor  # 실제값 -> 정규화값 변환
        self.depth_shift = depth_shift  # 최소값 offset
        self.use_depth_normalization = True
        
        # 기존 네트워크들...
        self.bbox_mlp = nn.Sequential(
            nn.Linear(bbox_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 출력을 [0,1] 범위로 제한
        )
        
        self.uv_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 1),
            nn.Sigmoid()  # 출력을 [0,1] 범위로 제한
        )
        
        # 나머지 네트워크들...
        # self.confidence_estimator = ConfidenceEstimator(input_dim=512)
        self.dynamic_weighter = DynamicWeightingNetwork(feature_dim=256)
        self.lidar_confidence_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(0.01, inplace=True),  # 음수 영역 0.01 기울기 유지
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        for layer in self.lidar_confidence_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        self.confidence_threshold = confidence_threshold
        
    def normalize_depth(self, depth_values):
        """실제 depth 값을 정규화된 값으로 변환"""
        # Clamp to reasonable range
        depth_clamped = torch.clamp(depth_values, min=0.1+1e-6, max=80.0-1e-6)
        # Normalize: (depth - min) / (max - min)  
        normalized = (depth_clamped - self.depth_shift) / (self.depth_scale_factor - self.depth_shift)
        return normalized
        
    def denormalize_depth(self, normalized_values):
        """정규화된 값을 실제 depth 값으로 변환"""
        # Denormalize: normalized * (max - min) + min
        depth_values = normalized_values * (self.depth_scale_factor - self.depth_shift) + self.depth_shift
        return torch.clamp(depth_values, min=0.1, max=80.0)
    
    # def normalize_depth(self, depth_values):
    #     """더 안정적인 정규화 함수"""
    #     # 클램핑 전 로그 변환으로 극단값 완화
    #     depth_clamped = torch.clamp(depth_values, min=0.1, max=80.0)
    #     depth_log = torch.log(depth_clamped + 1e-6)  # 로그 스케일링
    #     normalized = (depth_log - self.depth_shift) / (self.depth_scale_factor - self.depth_shift + 1e-6)
    #     return torch.clamp(normalized, 0.0, 1.0)  # 출력 범위 강제

    # def denormalize_depth(self, normalized_values):
    #     """역정규화 시 로그 복원"""
    #     normalized = torch.clamp(normalized_values, 0.0, 1.0)
    #     depth_log = normalized * (self.depth_scale_factor - self.depth_shift) + self.depth_shift
    #     return torch.exp(depth_log) - 1e-6  # 지수 복원
  
    def get_uv_indices(self, u, v, H, W):
        """UV coordinates를 embedding indices로 안전하게 변환"""
        # Hash function으로 indices를 제한된 범위로 매핑
        indices = (u.long() * 73856093 + v.long() * 19349663) % self.hash_embed_size
        return indices.long()
        
    def estimate_lidar_confidence(self, depth_values, uv_coords):
        assert not torch.isnan(uv_coords).any(), "NaN in UV coordinates"
        assert (depth_values >= 0).all(), "Negative depth values"
        """라이다 실측치의 신뢰도를 추정"""
        depth_valid = (depth_values > 0).float()
        
        # 정규화된 depth로 변환하여 confidence 계산
        # normalized_depth = self.normalize_depth(depth_values)
        normalized_depth = depth_values
        
        coord_feats = torch.cat([
            normalized_depth.unsqueeze(-1),  # 정규화된 depth 사용
            uv_coords[:, 2:3].float() / 1600.0,  # UV도 정규화
            uv_coords[:, 3:4].float() / 900.0  # UV도 정규화
        ], dim=-1)
        
        confidence = self.lidar_confidence_net(coord_feats)
        confidence = confidence * depth_valid.unsqueeze(-1)
        
        return confidence.squeeze(-1)
    
    # def forward(self, uv, depth_map, bbox_feats,training_stage='full'):
    #     """
    #     Args:
    #         uv: [N, 4] (cam_id, obj_id, u, v)
    #         depth_map: [num_cams, H, W] - 실제 LiDAR depth 값
    #         bbox_feats: [N, 256, 7, 7]
    #         training_stage: 'estimation_only', 'mixed', 'full'
    #     """
    #     N = uv.size(0)
    #     H, W = depth_map.shape[1], depth_map.shape[2]

    #     cam_ids = uv[:, 0].long()
    #     u = uv[:, 2].clamp(0, W-1).long()
    #     v = uv[:, 3].clamp(0, H-1).long()

    #     # 1. 라이다 depth 값 조회
    #     z_depth_real = depth_map[cam_ids, v, u]

    #     # 3. LiDAR depth를 정규화
    #     z_depth_normalized = self.normalize_depth(z_depth_real)
    #     # 4. 라이다 confidence 추정 (정규화된 값 사용)
    #     lidar_confidence = self.estimate_lidar_confidence(z_depth_normalized, uv)

    #     # 2. 추정값 계산 (기존 estimation 로직)
    #     indices = self.get_uv_indices(u, v, H, W)
    #     uv_embed = self.uv_embed(indices)
    #     z_uv = self.uv_mlp(uv_embed).squeeze(1)
    #     bbox_flat = bbox_feats.view(N, -1)
    #     z_bbox = self.bbox_mlp(bbox_flat).squeeze(1)
    #     alpha = torch.sigmoid(self.dynamic_weighter(uv_embed, bbox_feats.view(N, -1)[:, :256]).squeeze(1))

    #     z_estimated = alpha * z_uv + (1 - alpha) * z_bbox
    #     z_estimated = torch.clamp(z_estimated, 0.0, 1.0)  # [0,1] 범위 강제
    #     z_estimated_real = self.denormalize_depth(z_estimated)  # ref_points_uvz는 [N, 3] 형태로 가정

    #     # 3. 라이다 값이 유효하면 그대로, 아니면 estimation 사용
    #     valid_lidar_mask = (z_depth_real > 0)
    #     z_final_real = torch.where(valid_lidar_mask, z_depth_real, z_estimated_real)

    #     return {
    #         'depth': z_final_real.unsqueeze(-1),  # 최종 depth (라이다 or estimation)
    #         'z_lidar_real': z_depth_real,         # 라이다 실측값
    #         'confidence': lidar_confidence,       # 라이다 confidence
    #         'z_estimated_real': z_estimated_real, # 추정값
    #         # 'alpha': alpha,                       # dynamic weighting
    #         'lidar_mask': valid_lidar_mask,       # 라이다 사용 여부
    #     }
    
    def forward(self, uv, depth_map, bbox_feats, training_stage='full'):
        # ... (기존 코드 동일)
        N = uv.size(0)
        H, W = depth_map.shape[1], depth_map.shape[2]

        cam_ids = uv[:, 0].long()
        u = uv[:, 2].clamp(0, W-1).long()
        v = uv[:, 3].clamp(0, H-1).long()
        
        # 1. 라이다 depth 값 조회 (변경 없음)
        z_depth_real = depth_map[cam_ids, v, u]
        
        # 2. LiDAR 정규화 및 confidence 추정 (변경 없음)
        z_depth_normalized = self.normalize_depth(z_depth_real)
        lidar_confidence = self.estimate_lidar_confidence(z_depth_normalized, uv)
        
        # 3. 깊이 추정 계산 (변경 없음)
        indices = self.get_uv_indices(u, v, H, W)
        uv_embed = self.uv_embed(indices)
        z_uv = self.uv_mlp(uv_embed).squeeze(1)
        bbox_flat = bbox_feats.view(N, -1)
        z_bbox = self.bbox_mlp(bbox_flat).squeeze(1)
        alpha = torch.sigmoid(self.dynamic_weighter(uv_embed, bbox_feats.view(N, -1)[:, :256]).squeeze(1))
        
        z_estimated = alpha * z_uv + (1 - alpha) * z_bbox
        z_estimated = torch.clamp(z_estimated, 0.0, 1.0)
        z_estimated_real = self.denormalize_depth(z_estimated)
        
        # 4. 확률적 융합 적용 (핵심 변경)
        valid_lidar_mask = (z_depth_real > 0)
        lidar_confidence_adjusted = lidar_confidence * valid_lidar_mask.float()
        
        # 신뢰도 기반 가중 평균
        z_final_real = (
            lidar_confidence_adjusted * z_depth_real +
            (1 - lidar_confidence_adjusted) * z_estimated_real
        )
        
        return {
            'depth': z_final_real.unsqueeze(-1),
            'z_lidar_real': z_depth_real,
            'confidence': lidar_confidence,
            'z_estimated_real': z_estimated_real,
            'lidar_mask': valid_lidar_mask,
        }

    # def forward(self, uv, depth_map, bbox_feats, training_stage='full'):
    #     """
    #     Args:
    #         uv: [N, 4] (cam_id, obj_id, u, v)
    #         depth_map: [num_cams, H, W] - 실제 LiDAR depth 값
    #         bbox_feats: [N, 256, 7, 7]
    #         training_stage: 'estimation_only', 'mixed', 'full'
    #     """
    #     N = uv.size(0)
    #     H, W = depth_map.shape[1], depth_map.shape[2]

    #     # 1. 입력 분해
    #     cam_ids = uv[:, 0].long()
    #     u = uv[:, 2].clamp(0, W-1).long()
    #     v = uv[:, 3].clamp(0, H-1).long()

    #     # 2. 라이다 depth 값 조회 (실제값)
    #     z_depth_real = depth_map[cam_ids, v, u]  # 실제 LiDAR 값
        
    #     # 3. LiDAR depth를 정규화
    #     z_depth_normalized = self.normalize_depth(z_depth_real)
        
    #     # 4. 라이다 confidence 추정 (정규화된 값 사용)
    #     lidar_confidence = self.estimate_lidar_confidence(z_depth_normalized, uv)
        
    #     # 5. Feature extraction
    #     indices = self.get_uv_indices(u, v, H, W)
    #     uv_embed = self.uv_embed(indices)  # [N, 256]
    #     z_uv = self.uv_mlp(uv_embed).squeeze(1)  # [N] - 정규화된 출력
        
    #     # BBox features  
    #     bbox_flat = bbox_feats.view(N, -1)
    #     z_bbox = self.bbox_mlp(bbox_flat).squeeze(1)  # [N] - 정규화된 출력
        
    #     # 6. Dynamic weighting
    #     alpha = self.dynamic_weighter(uv_embed, 
    #                                 bbox_feats.view(N, -1)[:, :256]).squeeze(1)
        
    #     # 7. Confidence-based depth estimation (모든 값이 정규화된 상태)
    #     if training_stage == 'estimation_only':
    #         z_final_normalized = alpha * z_uv + (1 - alpha) * z_bbox
            
    #     elif training_stage == 'mixed':
    #         high_confidence_mask = (lidar_confidence > self.confidence_threshold) & (z_depth_real > 0)
    #         z_estimated_normalized = alpha * z_uv + (1 - alpha) * z_bbox
    #         z_final_normalized = torch.where(high_confidence_mask, z_depth_normalized, z_estimated_normalized)
            
    #     else:  # 'full'
    #         valid_lidar_mask = (z_depth_real > 0)
    #         z_estimated_normalized = alpha * z_uv + (1 - alpha) * z_bbox
            
    #         lidar_weight = lidar_confidence * valid_lidar_mask.float()
    #         z_final_normalized = lidar_weight * z_depth_normalized + (1 - lidar_weight) * z_estimated_normalized
        
    #     # 8. 최종 출력을 실제 스케일로 변환
    #     z_final_real = self.denormalize_depth(z_final_normalized)
        
    #     return {
    #         'depth': z_final_real.unsqueeze(-1),  # 실제 스케일 depth
    #         'depth_normalized': z_final_normalized.unsqueeze(-1),  # 정규화된 depth (디버깅용)
    #         'confidence': lidar_confidence,
    #         'alpha': alpha,
    #         'lidar_weight': lidar_confidence * (z_depth_real > 0).float() if training_stage == 'full' else None,
    #         'z_lidar_real': z_depth_real,  # 디버깅용
    #         'lidar_depth_normalized': z_depth_normalized  # 디버깅용
    #     }

# class pts_regressor(nn.Module) :
#     def __init__(self, dropout=0.0 , num_kp=100) :
#         super(pts_regressor,self).__init__()

#         self.num_kp = num_kp
#         self.leakyRELU = nn.LeakyReLU(0.1)
#         self.mish = nn.Mish()
        
#         # transformer encoder feature aggregration
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
#         self.flatten = nn.Flatten()
#         self.fc0_aggr = nn.Linear(self.num_kp*3 + 312 , 1024) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
#         self.bn0_aggr = nn.BatchNorm1d(1024)
#         self.fc = nn.Linear(1024, 300)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, corrs_pred, enc_out):
#         """
#         Args:
#             corr_pred: [B, N, 3] (N: number of keypoints)
#             enc_out: [B, C, H, W] (C: channel, H: height, W: width)
#         Returns:
#             pts_regressed: [B, N, 3]
#         """
#         # enc_out shape: [B, C, H, W]
#         pts_avg =self.avgpool(enc_out)
#         pts_avg = self.flatten(pts_avg)
#         corrs_emb = corrs_pred.view(6, -1)
#         feature_emb=torch.cat((pts_avg,corrs_emb),dim=-1)
#         pts_x = self.mish(self.fc0_aggr(feature_emb))
#         pts_x = self.bn0_aggr(pts_x)
#         pts_x = self.dropout(pts_x)
#         pts_x = self.fc(pts_x)
#         pts_x = pts_x.view(6,100,-1)
        
#         return pts_x

class SimplifiedDepthEstimator(nn.Module):
    def __init__(self, bbox_feat_dim=256*7*7, depth_scale_factor=80.0, depth_shift=0.1):
        super().__init__()
        
        # 단일 MLP로 단순화
        self.depth_predictor = nn.Sequential(
            nn.Linear(bbox_feat_dim + 2, 256),  # bbox_feat + uv coordinates
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0~1 범위로 제한
        )
        
        # **확률적 융합을 위한 LiDAR confidence 네트워크 추가**
        self.lidar_confidence_net = nn.Sequential(
            nn.Linear(3, 64),  # depth + normalized UV coordinates
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0~1 범위의 confidence
        )
        
        # Depth normalization parameters (기존 코드에서 가져옴)
        self.depth_scale_factor = depth_scale_factor
        self.depth_shift = depth_shift
        
        # 네트워크 초기화
        for layer in self.lidar_confidence_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
    
    def normalize_depth(self, depth_values):
        """실제 depth 값을 정규화된 값으로 변환"""
        depth_clamped = torch.clamp(depth_values, min=0.1+1e-6, max=80.0-1e-6)
        normalized = (depth_clamped - self.depth_shift) / (self.depth_scale_factor - self.depth_shift)
        return normalized
        
    def denormalize_depth(self, normalized_values):
        """정규화된 값을 실제 depth 값으로 변환"""
        depth_values = normalized_values * (self.depth_scale_factor - self.depth_shift) + self.depth_shift
        return torch.clamp(depth_values, min=0.1, max=80.0)
    
    def estimate_lidar_confidence(self, depth_values, uv_coords):
        """라이다 실측치의 신뢰도를 추정"""
        # NaN 및 음수 값 검증
        assert not torch.isnan(uv_coords).any(), "NaN in UV coordinates"
        assert (depth_values >= 0).all(), "Negative depth values"
        
        depth_valid = (depth_values > 0).float()
        
        # 정규화된 depth 사용
        normalized_depth = self.normalize_depth(depth_values)
        
        # UV 좌표도 정규화 (이미지 크기에 맞게)
        coord_feats = torch.cat([
            normalized_depth.unsqueeze(-1),
            uv_coords[:, 2:3].float() / 1600.0,  # u 좌표 정규화
            uv_coords[:, 3:4].float() / 900.0   # v 좌표 정규화
        ], dim=-1)  # [N, 3]
        
        confidence = self.lidar_confidence_net(coord_feats)
        confidence = confidence * depth_valid.unsqueeze(-1)
        
        return confidence.squeeze(-1)  # [N]
        
    def forward(self, uv, depth_map, bbox_feats):
        """
        Args:
            uv: [N, 4] (cam_id, obj_id, u, v)
            depth_map: [num_cams, H, W] 
            bbox_feats: [N, 256, 7, 7]
        Returns:
            dict with 'depth' and other info
        """
        N = uv.size(0)
        H, W = depth_map.shape[1], depth_map.shape[2]
        
        # 1. 직접적인 depth 조회
        cam_ids = uv[:, 0].long()
        u = uv[:, 2].clamp(0, W-1).long()
        v = uv[:, 3].clamp(0, H-1).long()
        
        z_depth_real = depth_map[cam_ids, v, u]  # 실제 LiDAR 값
        
        # 2. **LiDAR confidence 추정 (확률적 융합을 위해)**
        lidar_confidence = self.estimate_lidar_confidence(z_depth_real, uv)
        
        # 3. **Depth 추정 (depth가 없는 경우를 위해)**
        bbox_flat = bbox_feats.view(N, -1)
        uv_coords = uv[:, 2:4].float() / torch.tensor([W, H], device=uv.device)
        combined_input = torch.cat([bbox_flat, uv_coords], dim=1)
        
        z_estimated_normalized = self.depth_predictor(combined_input).squeeze(1)  # [N] (0~1)
        z_estimated_real = self.denormalize_depth(z_estimated_normalized)  # 실제 스케일로 변환
        
        # 4. **확률적 융합 적용 (핵심 변경)**
        valid_lidar_mask = (z_depth_real > 0)
        lidar_confidence_adjusted = lidar_confidence * valid_lidar_mask.float()
        
        # **신뢰도 기반 가중 평균**
        z_final_real = (
            lidar_confidence_adjusted * z_depth_real +
            (1 - lidar_confidence_adjusted) * z_estimated_real
        )
        
        return {
            'depth': z_final_real.unsqueeze(-1),           # 최종 depth 값 [N, 1]
            'z_lidar_real': z_depth_real,                  # 원본 LiDAR 값 [N]
            'confidence': lidar_confidence,                # LiDAR 신뢰도 [N]
            'z_estimated_real': z_estimated_real,          # 추정된 depth [N]
            'lidar_mask': valid_lidar_mask,               # 유효한 LiDAR 마스크 [N]
            'confidence_weights': lidar_confidence_adjusted  # 실제 적용된 가중치 [N]
        }


# class SelfSupervisedCorrespondenceLoss(nn.Module):
#     """
#     Camera-LiDAR Correspondence를 위한 Self-supervised Loss 클래스
#     GT correspondence 없이 geometric, photometric consistency 활용
#     """
    
#     def __init__(self, 
#                  cycle_weight=1.0, 
#                  photo_weight=0.3, 
#                  geom_weight=0.2, 
#                  smooth_weight=0.1, 
#                  range_weight=0.05,
#                  temporal_weight=0.3):
#         super(SelfSupervisedCorrespondenceLoss, self).__init__()
        
#         # Loss 가중치 설정
#         self.cycle_weight = cycle_weight
#         self.photo_weight = photo_weight
#         self.geom_weight = geom_weight
#         self.smooth_weight = smooth_weight
#         self.range_weight = range_weight
#         self.temporal_weight = temporal_weight
        
#         # 이전 프레임 정보 저장용
#         self.prev_pred = None
#         self.prev_query = None
        
#         # Edge detection을 위한 Sobel 필터
#         self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
#                                                     dtype=torch.float32).view(1, 1, 3, 3))
#         self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
#                                                     dtype=torch.float32).view(1, 1, 3, 3))

#     def compute_cycle_consistency(self, pred, query_points, corr_network=None):
#         """
#         개선된 Cycle Consistency - 역방향 네트워크 없이 구현
#         Args:
#             pred: [B, N, 2] 예측된 correspondence points
#             query_points: [B, N, 2] 원본 query points
#             corr_network: correspondence network (self-consistency용)
#         """
#         # Option 1: Self-consistency (같은 네트워크 사용)
#         if corr_network is not None:
#             # pred를 새로운 query로 사용하여 역방향 예측
#             reconstructed_query = corr_network(pred)  # 네트워크가 pred를 query로 처리
#             cycle_loss = F.mse_loss(reconstructed_query, query_points)
#         else:
#             # Simplified cycle consistency - identity mapping 가정
#             cycle_loss = F.mse_loss(pred, query_points)
        
#         # Option 2: Temporal consistency (연속 프레임 활용)
#         if self.prev_pred is not None and self.prev_query is not None:
#             temporal_cycle = F.mse_loss(
#                 pred - self.prev_pred, 
#                 query_points - self.prev_query
#             )
#             cycle_loss += self.temporal_weight * temporal_cycle
        
#         # 현재 값들을 다음 프레임을 위해 저장
#         self.prev_pred = pred.detach().clone()
#         self.prev_query = query_points.detach().clone()
        
#         return cycle_loss

#     def compute_photometric_consistency(self, pred, query_points, camera_img, depth_map):
#         """
#         실제 구현 가능한 Photometric Consistency
#         RGB intensity와 depth value 간의 구조적 유사성 평가
#         """
#         # Grid sampling을 이용한 feature 추출
#         query_rgb = F.grid_sample(
#             camera_img, 
#             query_points.unsqueeze(2) * 2 - 1,  # [0,1] -> [-1,1] 범위로 변환
#             align_corners=False,
#             mode='bilinear',
#             padding_mode='border'
#         ).squeeze(-1).transpose(1, 2)  # [B, N, 3]
        
#         pred_depth = F.grid_sample(
#             depth_map.unsqueeze(1), 
#             pred.unsqueeze(2) * 2 - 1,
#             align_corners=False,
#             mode='bilinear',
#             padding_mode='border'
#         ).squeeze(-1).transpose(1, 2)  # [B, N, 1]
        
#         # Structural similarity 기반 consistency
#         # RGB intensity와 depth value 간의 구조적 유사성 평가
#         rgb_intensity = torch.mean(query_rgb, dim=-1, keepdim=True)  # Grayscale conversion
        
#         # Normalization for better correlation
#         normalized_depth = (pred_depth - pred_depth.mean(dim=1, keepdim=True)) / (pred_depth.std(dim=1, keepdim=True) + 1e-6)
#         normalized_rgb = (rgb_intensity - rgb_intensity.mean(dim=1, keepdim=True)) / (rgb_intensity.std(dim=1, keepdim=True) + 1e-6)
        
#         # Cross-correlation 기반 consistency
#         photo_loss = 1.0 - F.cosine_similarity(
#             normalized_rgb.flatten(1), 
#             normalized_depth.flatten(1), 
#             dim=1
#         ).mean()
        
#         return photo_loss

#     def compute_geometric_consistency(self, pred, query_points, depth_map=None):
#         """
#         단순화된 Geometric Consistency
#         상대적 거리 보존 원칙 활용
#         """
#         B, N = query_points.shape[:2]
        
#         if N < 2:
#             return torch.tensor(0.0, device=pred.device)
        
#         # Query points 간 거리 계산
#         query_flat = query_points.view(B, N, 2)
#         query_dist = torch.cdist(query_flat, query_flat)  # [B, N, N]
        
#         # Predicted points 간 거리 계산  
#         pred_flat = pred.view(B, N, 2)
#         pred_dist = torch.cdist(pred_flat, pred_flat)  # [B, N, N]
        
#         # 거리 비율 보존 loss (대각선 제외)
#         mask = ~torch.eye(N, device=pred.device, dtype=torch.bool)
#         mask = mask.unsqueeze(0).expand(B, -1, -1)
        
#         query_dist_masked = query_dist[mask]
#         pred_dist_masked = pred_dist[mask]
        
#         # Normalize distances to prevent scale issues
#         query_dist_norm = query_dist_masked / (query_dist_masked.max() + 1e-6)
#         pred_dist_norm = pred_dist_masked / (pred_dist_masked.max() + 1e-6)
        
#         geom_loss = F.mse_loss(query_dist_norm, pred_dist_norm)
        
#         return geom_loss

#     def compute_spatial_smoothness(self, pred, query_points, camera_img):
#         """
#         Edge-aware spatial smoothness 적용
#         물체 경계에서는 smoothness 완화, 평면 영역에서는 강화
#         """
#         B, N = query_points.shape[:2]
        
#         if N < 2:
#             return torch.tensor(0.0, device=pred.device)
        
#         # 이미지 gradient 계산 (edge detection)
#         gray_img = torch.mean(camera_img, dim=1, keepdim=True)  # [B, 1, H, W]
        
#         # Sobel edge detection
#         grad_x = F.conv2d(gray_img, self.sobel_x, padding=1)
#         grad_y = F.conv2d(gray_img, self.sobel_y, padding=1)
#         edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
#         # Query points에서 edge strength 샘플링
#         edge_weight = F.grid_sample(
#             edge_magnitude,
#             query_points.unsqueeze(2) * 2 - 1,
#             align_corners=False,
#             mode='bilinear',
#             padding_mode='border'
#         ).squeeze(-1).squeeze(1)  # [B, N]
        
#         # Edge가 약한 영역에서만 smoothness 적용
#         query_diff = query_points[:, 1:] - query_points[:, :-1]  # [B, N-1, 2]
#         pred_diff = pred[:, 1:] - pred[:, :-1]  # [B, N-1, 2]
        
#         # Edge weight 기반 adaptive smoothness
#         edge_threshold = 0.1
#         edge_mask = (edge_weight[:, :-1] < edge_threshold).float()  # Low edge regions
        
#         # Smoothness loss with edge awareness
#         smooth_loss = torch.mean(
#             edge_mask.unsqueeze(-1) * (pred_diff - query_diff) ** 2
#         )
        
#         return smooth_loss

#     def compute_range_penalty(self, pred, x_range=(0.4, 1.0), y_range=(-0.1, 1.0)):
#         """
#         적응적 범위 제약 - 학습 초기에는 완화된 범위 적용
#         """
#         x_min, x_max = x_range
#         y_min, y_max = y_range
        
#         range_penalty = torch.mean(
#             0.1 * F.relu(x_min - pred[..., 0]) +  # x 좌표 하한
#             F.relu(pred[..., 0] - x_max) +        # x 좌표 상한
#             F.relu(y_min - pred[..., 1]) +        # y 좌표 하한  
#             F.relu(pred[..., 1] - y_max)          # y 좌표 상한
#         )
        
#         return range_penalty

#     def forward(self, pred, query_points, camera_img, depth_map, corr_network=None):
#         """
#         통합 Self-supervised Loss 계산
#         Args:
#             pred: [B, N, 2] 예측된 correspondence points
#             query_points: [B, N, 2] 원본 query points  
#             camera_img: [B, 3, H, W] 카메라 이미지
#             depth_map: [B, H, W] LiDAR depth map
#             corr_network: correspondence network (optional)
#         """
#         # 1. Cycle Consistency Loss
#         cycle_loss = self.compute_cycle_consistency(pred, query_points, corr_network)
        
#         # 2. Photometric Consistency Loss
#         photo_loss = self.compute_photometric_consistency(pred, query_points, camera_img, depth_map)
        
#         # 3. Geometric Consistency Loss
#         geom_loss = self.compute_geometric_consistency(pred, query_points, depth_map)
        
#         # 4. Edge-aware Spatial Smoothness Loss
#         smooth_loss = self.compute_spatial_smoothness(pred, query_points, camera_img)
        
#         # 5. Adaptive Range Penalty
#         range_penalty = self.compute_range_penalty(pred)
        
#         # 가중 합산
#         total_loss = (self.cycle_weight * cycle_loss + 
#                      self.photo_weight * photo_loss + 
#                      self.geom_weight * geom_loss + 
#                      self.smooth_weight * smooth_loss + 
#                      self.range_weight * range_penalty)
        
#         # 상세 손실 정보 반환
#         loss_dict = {
#             'total': total_loss.item(),
#             'cycle': cycle_loss.item(),
#             'photo': photo_loss.item(), 
#             'geom': geom_loss.item(),
#             'smooth': smooth_loss.item(),
#             'range': range_penalty.item()
#         }
        
#         return total_loss, loss_dict

#     def update_weights(self, epoch, total_epochs):
#         """
#         학습 단계에 따른 가중치 적응적 조정
#         """
#         progress = epoch / total_epochs
        
#         if progress < 0.3:  # 초기 단계: cycle consistency 중심
#             self.cycle_weight = 1.0
#             self.photo_weight = 0.1
#             self.geom_weight = 0.1
#         elif progress < 0.7:  # 중간 단계: photometric consistency 추가
#             self.cycle_weight = 0.8
#             self.photo_weight = 0.3
#             self.geom_weight = 0.2
#         else:  # 후기 단계: 모든 loss 균형
#             self.cycle_weight = 0.6
#             self.photo_weight = 0.3
#             self.geom_weight = 0.2

#     def reset_temporal_memory(self):
#         """
#         새로운 시퀀스 시작 시 temporal consistency 메모리 초기화
#         """
#         self.prev_pred = None
#         self.prev_query = None


class SelfSupervisedCorrespondenceLoss(nn.Module):
    """
    Camera-LiDAR Correspondence를 위한 Self-supervised Loss 클래스
    GT correspondence 없이 geometric, photometric consistency 활용
    """
    
    def __init__(self, 
                 cycle_weight=1.0, 
                 photo_weight=0.3, 
                 geom_weight=0.2, 
                 smooth_weight=0.1, 
                 range_weight=0.05,
                 temporal_weight=0.3):
        super(SelfSupervisedCorrespondenceLoss, self).__init__()
        
        # Loss 가중치 설정
        self.cycle_weight = cycle_weight
        self.photo_weight = photo_weight
        self.geom_weight = geom_weight
        self.smooth_weight = smooth_weight
        self.range_weight = range_weight
        self.temporal_weight = temporal_weight
        
        # 이전 프레임 정보 저장용
        self.prev_pred = None
        self.prev_query = None
        
        # Edge detection을 위한 Sobel 필터
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))

    # def compute_cycle_consistency(self, pred, query_points, corr_network=None):
    #     """
    #     개선된 Cycle Consistency - 역방향 네트워크 없이 구현
    #     Args:
    #         pred: [B, N, 2] 예측된 correspondence points
    #         query_points: [B, N, 2] 원본 query points
    #         corr_network: correspondence network (self-consistency용)
    #     """
    #     # Option 1: Self-consistency (같은 네트워크 사용)
    #     if corr_network is not None:
    #         # pred를 새로운 query로 사용하여 역방향 예측
    #         reconstructed_query = corr_network(pred)  # 네트워크가 pred를 query로 처리
    #         cycle_loss = F.mse_loss(reconstructed_query, query_points)
    #     else:
    #         # Simplified cycle consistency - identity mapping 가정
    #         cycle_loss = F.mse_loss(pred, query_points)
        
    #     # Option 2: Temporal consistency (연속 프레임 활용)
    #     if self.prev_pred is not None and self.prev_query is not None:
    #         temporal_cycle = F.mse_loss(
    #             pred - self.prev_pred, 
    #             query_points - self.prev_query
    #         )
    #         cycle_loss += self.temporal_weight * temporal_cycle
        
    #     # 현재 값들을 다음 프레임을 위해 저장
    #     self.prev_pred = pred.detach().clone()
    #     self.prev_query = query_points.detach().clone()
        
    #     return cycle_loss
    
    def compute_cycle_consistency(self, pred, query_points):
        """
        실제 구현 가능한 cycle consistency
        u: [0.5,1] (pred), [0,0.5] (query)
        v: [0,1] (both)
        """
        # Option 1: Range-aware identity mapping (u 좌표만 변환)
        # pred_u: [0.5,1] -> [-1,1], query_u: [0,0.5] -> [-1,1]
        u_pred_normalized = (pred[..., 0] - 0.75) / 0.25
        u_query_normalized = query_points[..., 0] * 4 - 1
        
        # v 좌표는 동일 범위 [0,1]이므로 정규화 없이 직접 비교
        v_pred = pred[..., 1]
        v_query = query_points[..., 1]
        
        # u와 v를 별도로 계산 후 합산
        u_loss = F.mse_loss(u_pred_normalized, u_query_normalized)
        v_loss = F.mse_loss(v_pred, v_query)
        range_loss = (u_loss + v_loss) / 2.0

        # Option 2: Contrastive learning 기반 (u 좌표만 offset 적용)
        # u: query + 0.5 -> [0.5,1] 범위로 변환
        # v: 동일 범위 유지
        contrastive_points = query_points.clone()
        contrastive_points[..., 0] += 0.5  # u 좌표만 0.5 이동
        
        # Cosine 유사도 계산 (차원 유지)
        similarity = F.cosine_similarity(pred, contrastive_points, dim=-1)
        contrastive_loss = 1.0 - similarity.mean()

        # 두 옵션을 결합 (가중치 조정 가능)
        cycle_loss = 0.7 * range_loss + 0.3 * contrastive_loss
        
        return cycle_loss


    def compute_photometric_consistency(self, pred, query_points, camera_img, depth_map):
        """
        실제 구현 가능한 Photometric Consistency
        RGB intensity와 depth value 간의 구조적 유사성 평가
        """
        # Grid sampling을 이용한 feature 추출
        query_rgb = F.grid_sample(
            camera_img, 
            query_points.unsqueeze(2) * 2 - 1,  # [0,1] -> [-1,1] 범위로 변환
            align_corners=False,
            mode='bilinear',
            padding_mode='border'
        ).squeeze(-1).transpose(1, 2)  # [B, N, 3]
        
        pred_depth = F.grid_sample(
            depth_map.unsqueeze(1), 
            pred.unsqueeze(2) * 2 - 1,
            align_corners=False,
            mode='bilinear',
            padding_mode='border'
        ).squeeze(-1).transpose(1, 2)  # [B, N, 1]
        
        # Structural similarity 기반 consistency
        # RGB intensity와 depth value 간의 구조적 유사성 평가
        rgb_intensity = torch.mean(query_rgb, dim=-1, keepdim=True)  # Grayscale conversion
        
        # Normalization for better correlation
        normalized_depth = (pred_depth - pred_depth.mean(dim=1, keepdim=True)) / (pred_depth.std(dim=1, keepdim=True) + 1e-6)
        normalized_rgb = (rgb_intensity - rgb_intensity.mean(dim=1, keepdim=True)) / (rgb_intensity.std(dim=1, keepdim=True) + 1e-6)
        
        # Cross-correlation 기반 consistency
        photo_loss = 1.0 - F.cosine_similarity(
            normalized_rgb.flatten(1), 
            normalized_depth.flatten(1), 
            dim=1
        ).mean()
        
        return photo_loss

    def compute_geometric_consistency(self, pred, query_points, depth_map=None):
        """
        단순화된 Geometric Consistency
        상대적 거리 보존 원칙 활용
        """
        B, N = query_points.shape[:2]
        
        if N < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Query points 간 거리 계산
        query_flat = query_points.view(B, N, 2)
        query_dist = torch.cdist(query_flat, query_flat)  # [B, N, N]
        
        # Predicted points 간 거리 계산  
        pred_flat = pred.view(B, N, 2)
        pred_dist = torch.cdist(pred_flat, pred_flat)  # [B, N, N]
        
        # 거리 비율 보존 loss (대각선 제외)
        mask = ~torch.eye(N, device=pred.device, dtype=torch.bool)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        
        query_dist_masked = query_dist[mask]
        pred_dist_masked = pred_dist[mask]
        
        # Normalize distances to prevent scale issues
        query_dist_norm = query_dist_masked / (query_dist_masked.max() + 1e-6)
        pred_dist_norm = pred_dist_masked / (pred_dist_masked.max() + 1e-6)
        
        geom_loss = F.mse_loss(query_dist_norm, pred_dist_norm)
        
        return geom_loss

    def compute_spatial_smoothness(self, pred, query_points, camera_img):
        """
        Edge-aware spatial smoothness 적용
        물체 경계에서는 smoothness 완화, 평면 영역에서는 강화
        """
        B, N = query_points.shape[:2]
        
        if N < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # 이미지 gradient 계산 (edge detection)
        gray_img = torch.mean(camera_img, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Sobel edge detection
        grad_x = F.conv2d(gray_img, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray_img, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Query points에서 edge strength 샘플링
        edge_weight = F.grid_sample(
            edge_magnitude,
            query_points.unsqueeze(2) * 2 - 1,
            align_corners=False,
            mode='bilinear',
            padding_mode='border'
        ).squeeze(-1).squeeze(1)  # [B, N]
        
        # Edge가 약한 영역에서만 smoothness 적용
        query_diff = query_points[:, 1:] - query_points[:, :-1]  # [B, N-1, 2]
        pred_diff = pred[:, 1:] - pred[:, :-1]  # [B, N-1, 2]
        
        # Edge weight 기반 adaptive smoothness
        edge_threshold = 0.1
        edge_mask = (edge_weight[:, :-1] < edge_threshold).float()  # Low edge regions
        
        # Smoothness loss with edge awareness
        smooth_loss = torch.mean(
            edge_mask.unsqueeze(-1) * (pred_diff - query_diff) ** 2
        )
        
        return smooth_loss

    def compute_range_penalty(self, pred, x_range=(0.4, 1.0), y_range=(-0.1, 1.0)):
        """
        적응적 범위 제약 - 학습 초기에는 완화된 범위 적용
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        range_penalty = torch.mean(
            0.1 * F.relu(x_min - pred[..., 0]) +  # x 좌표 하한
            F.relu(pred[..., 0] - x_max) +        # x 좌표 상한
            F.relu(y_min - pred[..., 1]) +        # y 좌표 하한  
            F.relu(pred[..., 1] - y_max)          # y 좌표 상한
        )
        
        return range_penalty

    def forward(self, pred, query_points, camera_img, depth_map, corr_network=None):
        """
        통합 Self-supervised Loss 계산
        Args:
            pred: [B, N, 2] 예측된 correspondence points
            query_points: [B, N, 2] 원본 query points  
            camera_img: [B, 3, H, W] 카메라 이미지
            depth_map: [B, H, W] LiDAR depth map
            corr_network: correspondence network (optional)
        """
        # 1. Cycle Consistency Loss
        cycle_loss = self.compute_cycle_consistency(pred, query_points)
        
        # 2. Photometric Consistency Loss
        photo_loss = self.compute_photometric_consistency(pred, query_points, camera_img, depth_map)
        
        # 3. Geometric Consistency Loss
        geom_loss = self.compute_geometric_consistency(pred, query_points, depth_map)
        
        # 4. Edge-aware Spatial Smoothness Loss
        smooth_loss = self.compute_spatial_smoothness(pred, query_points, camera_img)
        
        # 5. Adaptive Range Penalty
        range_penalty = self.compute_range_penalty(pred)
        
        # 가중 합산
        total_loss = (self.cycle_weight * cycle_loss + 
                     self.photo_weight * photo_loss + 
                     self.geom_weight * geom_loss + 
                     self.smooth_weight * smooth_loss + 
                     self.range_weight * range_penalty)
        
        # 상세 손실 정보 반환
        loss_dict = {
            'total': total_loss.item(),
            'cycle': cycle_loss.item(),
            'photo': photo_loss.item(), 
            'geom': geom_loss.item(),
            'smooth': smooth_loss.item(),
            'range': range_penalty.item()
        }
        
        return total_loss, loss_dict

    def update_weights(self, epoch, total_epochs):
        """
        학습 단계에 따른 가중치 적응적 조정
        """
        progress = epoch / total_epochs
        
        if progress < 0.3:  # 초기 단계: cycle consistency 중심
            self.cycle_weight = 1.0
            self.photo_weight = 0.1
            self.geom_weight = 0.1
        elif progress < 0.7:  # 중간 단계: photometric consistency 추가
            self.cycle_weight = 0.8
            self.photo_weight = 0.3
            self.geom_weight = 0.2
        else:  # 후기 단계: 모든 loss 균형
            self.cycle_weight = 0.6
            self.photo_weight = 0.3
            self.geom_weight = 0.2

    def reset_temporal_memory(self):
        """
        새로운 시퀀스 시작 시 temporal consistency 메모리 초기화
        """
        self.prev_pred = None
        self.prev_query = None

    def get_loss_weights(self):
        """
        현재 loss 가중치 반환
        """
        return {
            'cycle_weight': self.cycle_weight,
            'photo_weight': self.photo_weight,
            'geom_weight': self.geom_weight,
            'smooth_weight': self.smooth_weight,
            'range_weight': self.range_weight,
            'temporal_weight': self.temporal_weight
        }

    def set_loss_weights(self, weights_dict):
        """
        Loss 가중치 수동 설정
        """
        for key, value in weights_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def debug_loss_components(self, pred, query_points, camera_img, depth_map):
        """
        각 loss 성분별 디버깅 정보 출력
        """
        with torch.no_grad():
            cycle_loss = self.compute_cycle_consistency(pred, query_points)
            photo_loss = self.compute_photometric_consistency(pred, query_points, camera_img, depth_map)
            geom_loss = self.compute_geometric_consistency(pred, query_points, depth_map)
            smooth_loss = self.compute_spatial_smoothness(pred, query_points, camera_img)
            range_penalty = self.compute_range_penalty(pred)
            
            print("=== Loss Components Debug ===")
            print(f"Cycle Loss: {cycle_loss.item():.6f}")
            print(f"Photo Loss: {photo_loss.item():.6f}")
            print(f"Geom Loss: {geom_loss.item():.6f}")
            print(f"Smooth Loss: {smooth_loss.item():.6f}")
            print(f"Range Penalty: {range_penalty.item():.6f}")
            
            # 예측값 분포 분석
            pred_x_mean = pred[..., 0].mean().item()
            pred_y_mean = pred[..., 1].mean().item()
            pred_x_std = pred[..., 0].std().item()
            pred_y_std = pred[..., 1].std().item()
            
            print(f"Pred X: mean={pred_x_mean:.4f}, std={pred_x_std:.4f}")
            print(f"Pred Y: mean={pred_y_mean:.4f}, std={pred_y_std:.4f}")
            
            # Query 분포 분석
            query_x_mean = query_points[..., 0].mean().item()
            query_y_mean = query_points[..., 1].mean().item()
            query_x_std = query_points[..., 0].std().item()
            query_y_std = query_points[..., 1].std().item()
            
            print(f"Query X: mean={query_x_mean:.4f}, std={query_x_std:.4f}")
            print(f"Query Y: mean={query_y_mean:.4f}, std={query_y_std:.4f}")
            print("============================")

from scipy.spatial import cKDTree

class GraphBEVLocalAlignNet(nn.Module):
    def __init__(self, k_neighbors=16, feature_dim=64):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.feature_dim = feature_dim

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.refine_net = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.offset_regressor = nn.Linear(feature_dim, 1)

    def forward(self, coarse_pose, depth_map):
        """
        Args:
            coarse_pose: [num_cams, num_pts, 2]  # (6, 200, 2)
            depth_map: [num_cams, H, W]          # (6, 900, 1600)
        Returns:
            fine_aligned_pose: [num_cams, num_pts, 3]  # (6, 200, 3)
        """
        num_cams, num_pts, _ = coarse_pose.shape
        device = coarse_pose.device
        H, W = depth_map.shape[1], depth_map.shape[2]

        fine_aligned_pose = []

        for cam in range(num_cams):
            # [num_pts, 2]
            points_2d = coarse_pose[cam].detach().cpu().numpy()
            # k-NN 그래프
            if points_2d.ndim == 1:
                points_2d = points_2d.reshape(-1, 2)
            kdtree = cKDTree(points_2d)
            _, neighbor_idx = kdtree.query(points_2d, k=self.k_neighbors)  # [num_pts, k]

            # 픽셀 좌표로 변환
            px = (coarse_pose[cam, :, 0] * (W - 1)).long().clamp(0, W - 1)
            py = (coarse_pose[cam, :, 1] * (H - 1)).long().clamp(0, H - 1)
            # [num_pts, 1, 1, 1]
            depth_vals = depth_map[cam, py, px].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            depth_encoded = self.depth_encoder(depth_vals)  # [num_pts, feature_dim, 1, 1]

            # 이웃 depth
            neighbor_depths = []
            for i in range(num_pts):
                idx = neighbor_idx[i]
                n_px = (coarse_pose[cam, idx, 0] * (W - 1)).long().clamp(0, W - 1)
                n_py = (coarse_pose[cam, idx, 1] * (H - 1)).long().clamp(0, H - 1)
                n_depth = depth_map[cam, n_py, n_px].unsqueeze(0)  # [1, k]
                neighbor_depths.append(n_depth)
            neighbor_depths = torch.cat(neighbor_depths, dim=0).to(device)  # [num_pts, k]
            neighbor_depths = neighbor_depths.unsqueeze(1).unsqueeze(3)  # [num_pts,1,k,1]
            neighbor_depth_encoded = self.depth_encoder(neighbor_depths)  # [num_pts, feature_dim, k, 1]

            # Local feature refinement
            depth_encoded_exp = depth_encoded.expand(-1, -1, self.k_neighbors, -1)  # [num_pts, feature_dim, k, 1]
            dual_depth = torch.cat([depth_encoded_exp, neighbor_depth_encoded], dim=1)  # [num_pts, 2*feature_dim, k, 1]
            refined_feat = self.refine_net(dual_depth)  # [num_pts, feature_dim, k, 1]

            # Offset regression
            refined_feat_mean = refined_feat.mean(dim=2).squeeze(-1)  # [num_pts, feature_dim]
            offset = self.offset_regressor(refined_feat_mean).squeeze(-1)  # [num_pts]

            # z값 보정 및 결과 저장
            z = depth_map[cam, py, px].float() + offset  # [num_pts]
            # [num_pts, 3] = [x, y, z]
            pose_xyz = torch.cat([coarse_pose[cam], z.unsqueeze(-1)], dim=-1)
            fine_aligned_pose.append(pose_xyz)

        fine_aligned_pose = torch.stack(fine_aligned_pose, dim=0)  # [num_cams, num_pts, 3]
        return fine_aligned_pose
    

# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
import os

from networkx import is_dominating_set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS 
from .mv2d_head import MV2DHead
from COTR.COTR_models.cotr_model_moon_Ver12_0 import build
from torchvision.transforms import functional as tvtf
from torchvision.ops import DeformConv2d
import easydict
from collections import defaultdict

cotr_args = easydict.EasyDict({
                "out_dir" : "general_config['out']",
                # "load_weights" : "None",
#                 "load_weights_path" : './COTR/out/default/checkpoint.pth.tar' ,
                # "load_weights_path" : "./models/200_checkpoint.pth.tar",
                "load_weights_path" : None,
                "load_weights_freeze" : False ,
                "max_corrs" : 1000 ,
                "dim_feedforward" : 1024 , 
                "backbone" : "resnet50" ,
                "hidden_dim" : 312 ,
                # "hidden_dim" : 136 ,
                "dilation" : False ,
                "dropout" : 0.1 ,
                "nheads" : 8 ,
                "layer" : "layer3" ,
                "enc_layers" : 6 ,
                "dec_layers" : 6 ,
                "position_embedding" : "lin_sine"
                
})
# from torchvision.models import resnet50
from image_processing_unit_Ver15_0 import (find_all_depthmap_z_adv,find_rois_nonzero_z,find_rois_nonzero_z_adv,find_rois_nonzero_z_adv1,
                                           find_rois_nonzero_z_adv2,find_rois_nonzero_z_adv3,find_rois_nonzero_z_adv4,find_rois_nonzero_z_adv5,
                                           find_rois_nonzero_z_adv6,find_rois_nonzero_z_adv8,find_rois_nonzero_z_adv9,differentiable_find_rois1,
                                           find_rois_nonzero_z_adv10,find_rois_nonzero_z_adv11,
                                           image_to_lidar_global,image_to_lidar_global_modi,image_to_lidar_global_modi1,image_to_lidar_global_modi3,
                                           lidar_to_image_with_index,diff_lidar_to_image_with_index,
                                           miscalib_transform, miscalib_transform1,miscalib_transform2,
                                           points2depthmap,dense_map_gpu_optimized,colormap,dense_map_from_depth_batch,batch_colormap,differentiable_colormap,
                                           two_images_side_by_side,two_images_side_by_side_gpu,transform_uv_points,
                                           display_depth_maps,scale_uvz_points,descale_uvz_points,
                                           normalize_uvz_points,normalize_uv_points,denormalize_uv_points,
                                           inverse_scale_uvz_points,
                                           trim_corrs,batched_trim_corrs,denormalize_points,process_queries,process_queries_adv,process_queries_adv_modified,
                                           process_queries_adv1,differentiable_process_queries,
                                           selected_image_to_lidar_global,
                                           pixel_to_normalized,center2lidar_batch,differentiable_center2lidar,
                                           project_lidar_to_image,minmax_normalize_uvz,minmax_denormalize_uvz,
                                           draw_corrs , lidar_to_image_no_filter,visualize_bboxes,
                                           geometric_propagation,trim_or_generate_points,
                                           deduplicate_obj_ids,merge_point_clouds,differentiable_deduplicate,differentiable_object_matching,
                                           differentiable_object_matching,differentiable_merge_point_clouds,
                                           convert_to_bbox_coordinates_matched, get_center_points,
                                           batch_rois_center_by_cam_id,remove_duplicate_objs)

# @CALIB_TRANSFORMER.register_module()
# class CalibTransformer(PETRTransformer):

#     def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
#         super(CalibTransformer, self).__init__(
#             encoder=encoder,
#             decoder=decoder,
#             init_cfg=init_cfg,
#             cross=cross
#         )

#     def forward(self, x, query_embed, mask, pos_embed, attn_mask=None, cross_attn_mask=None, **kwargs):
        
#         # x: [bs, n, c, h, w], mask: [bs, n, h, w], query_embed: [bs, n_query, c]
#         bs, n, c, h, w = x.shape
#         memory = x.permute(1, 3, 4, 0, 2).reshape(n * h * w, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
#         mask = mask.view(bs, n * h * w)  # [bs, n, h, w] -> [bs, n*h*w]
#         query_embed = query_embed.permute(1, 0, 2)
#         pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(n * h * w, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
#         target = torch.zeros_like(query_embed)
#         if cross_attn_mask is not None:
#             cross_attn_mask = cross_attn_mask.flatten(1, 3)   # [n_query, n, h, w] -> [n_query, n * h * w]

#         # out_dec: [num_layers, num_query, bs, dim]
#         out_dec = self.decoder(
#             query=target,
#             key=memory,
#             value=memory,
#             key_pos=pos_embed,
#             query_pos=query_embed,
#             key_padding_mask=mask,
#             attn_masks=[attn_mask, cross_attn_mask],
#             **kwargs,
#             )
#         out_dec = out_dec.transpose(1, 2)
#         memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
#         return out_dec, memory
    
# class CorrelationCycleLoss(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         super().__init__()
#         self.loss_weight = loss_weight

    # # def forward(self, corr_pred, corr_target, cycle, queries, mask):
    # def forward(self, corr_pred, corr_target):
    #     corr_loss = torch.nn.functional.mse_loss(corr_pred, corr_target)
        
    #     # if mask.sum() > 0:
    #     #     cycle_loss = torch.nn.functional.mse_loss(cycle[mask], queries[mask])
    #     #     corr_loss += cycle_loss 
        
    #     return self.loss_weight * corr_loss
    
    # def forward(self, corr_pred, corr_target):
    #     point_clouds_loss = torch.tensor([0.0]).to(corr_pred.device)
    #     error = (corr_pred - corr_target).norm(dim=0)
    #     error.clamp(100.)
    #     point_clouds_loss += error.mean()
    #     return self.loss_weight * (point_clouds_loss/corr_target.shape[0])

    # def forward(self, corr_pred, corr_target):
    #     def chamfer_distance(x, y):
    #         # 입력 텐서가 2차원인 경우 3차원으로 확장
    #         if x.dim() == 2:
    #             x = x.unsqueeze(0)
    #         if y.dim() == 2:
    #             y = y.unsqueeze(0)
    #         xx = torch.bmm(x, x.transpose(2, 1))
    #         yy = torch.bmm(y, y.transpose(2, 1))
    #         zz = torch.bmm(x, y.transpose(2, 1))
    #         diag_ind = torch.arange(0, x.size(1)).to(x.device)
    #         rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    #         ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    #         P = (rx.transpose(2, 1) + ry - 2 * zz)
    #         P_clamp = torch.clamp(P, min=0 ,max=100)
    #         # return P.min(1)[0].mean() + P.min(2)[0].mean()
    #         return P_clamp.min(1)[0].mean() + P_clamp.min(2)[0].mean()

    #     chamfer_loss = chamfer_distance(corr_pred, corr_target)
    #     return self.loss_weight * chamfer_loss / corr_target.shape[0]

class COTR(nn.Module):
    
    def __init__(self, num_kp=500):
        super(COTR, self).__init__()
        self.num_kp = num_kp
        ##### CORR network #######
        self.corr = build(cotr_args)
        # 배치 정규화 레이어 추가 (최종 출력 차원 기준)
        # self.final_bn = nn.BatchNorm1d(3)  # corrs_pred의 마지막 차원이 3인 경우
    
    def forward(self, sbs_img , query_input):

        for i in range(6) :
            # multi camera batch cotr 필요
            corrs_pred , enc_out = self.corr(sbs_img, query_input)

            # # 최종 출력 직전 배치 정규화 적용 (3D → 2D 변환)
            # B, N, C = corrs_pred.shape
            # corrs_pred = self.final_bn(
            #     corrs_pred.view(-1, C)  # (B*N, C) 형태로 평탄화
            # ).view(B, N, C)  # 원래 차원 복원

            img_reverse_input = torch.cat([sbs_img[..., 640:], sbs_img[..., :640]], axis=-1)
            ##cyclic loss pre-processing
            query_reverse = corrs_pred
            query_reverse[..., 0] = query_reverse[..., 0] - 0.5
            cycle,_ = self.corr(img_reverse_input, query_reverse)
            cycle[..., 0] = cycle[..., 0] - 0.5
            mask = torch.norm(cycle - query_input, dim=-1) < 40 / 640 # 40 pixel 거리에서는 마스크 

        return corrs_pred , cycle , mask , enc_out
    

class CorrelationCycleLoss(nn.Module):
    def __init__(self, corr_weight=1.0 , cycle_weight=1.0):
        super().__init__()
        self.corr_weight = corr_weight
        self.cycle_weight= cycle_weight

    def forward(self, corr_pred, corr_target, cycle, queries, mask):
        corr_loss = torch.nn.functional.mse_loss(corr_pred, corr_target)
        # Smooth L1 Loss 사용
        # corr_loss = torch.nn.functional.smooth_l1_loss(corr_pred, corr_target)
        cycle_loss = torch.tensor(0.0, device=corr_loss.device)
        
        if mask.sum() > 0:
            cycle_loss = torch.nn.functional.mse_loss(cycle[mask], queries[mask])
            # cycle_loss = torch.nn.functional.smooth_l1_loss(cycle[mask], queries[mask])
            corr_loss += cycle_loss 

        # return self.loss_weight * corr_loss
        return self.corr_weight * corr_loss + self.cycle_weight * cycle_loss

class PointDistanceLoss(nn.Module):
    def __init__(self, distance_weight=1.0):
        super().__init__()
        self.point_distance_weight = distance_weight
    
    def forward(self, points_pred, points_gt):
        return self.point_distance_loss(points_pred, points_gt) * self.point_distance_weight
    
    def chamfer_loss(self, points_a, points_b):
        """
        Chamfer Distance Loss 계산 메서드
        Args:
            points_a: (N, 3) 형태의 텐서 [detection_xyz_normal[...,2:]]
            points_b: (M, 3) 형태의 텐서 [pts_lidar_mis_normalized[mask_valid_mis]]
        """
        # 입력 차원 검증
        if points_a.size(0) == 0 or points_b.size(0) == 0:
            print ("chmfer loss points_a or points_b is empty") 
        assert points_a.dim() == 2 and points_b.dim() == 2, "Input must be 2D tensors"
        points_b = points_b.float()
        # 유효 포인트 필터링
        valid_a = torch.isfinite(points_a).all(dim=1)
        valid_b = torch.isfinite(points_b).all(dim=1)
        points_a = points_a[valid_a]
        points_b = points_b[valid_b]

        # 거리 행렬 계산
        dist_matrix = torch.cdist(points_a, points_b, p=2)
        
        # 양방향 최소 거리 계산
        min_a_to_b = torch.min(dist_matrix, dim=1)[0]
        min_b_to_a = torch.min(dist_matrix, dim=0)[0]
        
        # 평균 손실 계산
        return (min_a_to_b.mean() + min_b_to_a.mean()) / 2.0
    
    # def point_distance_loss(self, points_pred, points_gt):
    #     """
    #     1:1 대응 포인트 거리 손실 계산
    #     - points_a와 points_b는 (N, 3) 형태이며 동일한 개수의 포인트를 가져야 함
    #     - 각 포인트 쌍 간의 L2 거리 평균 계산
    #     """
    #     if points_pred.size(0) == 0 or points_gt.size(0) == 0:
    #         print("point_distance_loss: 입력 포인트 클라우드가 비어 있음")
    #         return torch.tensor(0.0, device=points_pred.device)
         
    #     assert points_pred.size() == points_gt.size(), "포인트 개수가 일치하지 않습니다"
    #     point_clouds_loss = torch.tensor([0.0]).to(points_pred.device)
    #     error = (points_pred - points_gt).norm(dim=0)
    #     error.clamp(100.)
    #     point_clouds_loss += error.mean()

    #     return point_clouds_loss/points_pred.shape[0]
    
    def point_distance_loss(self, points_pred, points_gt):
        """
        1:1 대응 포인트 거리 손실 계산
        - points_pred: [N,3], points_gt: [N,3]
        """
        if points_pred.size(0) == 0 or points_gt.size(0) == 0:
            return torch.tensor(0.0, device=points_pred.device)
        
        assert points_pred.size() == points_gt.size(), "포인트 개수 불일치"
        
        # 각 포인트별 L2 거리 계산 → [N]
        error = torch.norm(points_pred - points_gt, p=2, dim=1)
        
        # 값 클램핑 (100 이하로 제한)
        error = error.clamp(max=100.0)
        
        # 평균 손실 계산
        return error.mean()

    
# # Deformable SPN 모듈 정의
# class DeformableSPN(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super().__init__()
#         self.offset_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 18, 3, padding=1)  # 3x3 커널 기준 2*9 offset
#         )
#         self.dcn = DeformConv2d(in_channels, out_channels, 3, padding=1)

#     def forward(self, x):
#         offsets = self.offset_conv(x)
#         return self.dcn(x, offsets)
    
# class SkipConnectionNetwork(nn.Module):
#     def __init__(self, cam_feature_shape=(64, 312, 12), lidar_feature_dim=3, output_dim=3):
#         super().__init__()
#         self.cam_channel, self.cam_height, self.cam_width = cam_feature_shape
        
#         # 카메라 특징 처리 (공간 정보 집계)
#         self.cam_encoder = nn.Sequential(
#             nn.Linear(self.cam_channel, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 64)
#         )
        
#         # 라이다 특징 처리 (정규 좌표 정보)
#         self.lidar_encoder = nn.Sequential(
#             nn.Linear(lidar_feature_dim, 64),
#             nn.LayerNorm(64),
#             nn.ReLU()
#         )
        
#         # 융합 레이어 (교차 모달리티 학습)
#         self.fusion = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )

#     def forward(self, detection_xyz_normal, enc_out):
#         num_objects = detection_xyz_normal.size(0)
#         num_cameras = enc_out.size(0)
        
#         # 카메라 특징 처리 파이프라인
#         cam_features = enc_out.contiguous().reshape(num_cameras, self.cam_channel, -1)  # [C, 64, 312*12]
#         cam_features = cam_features.permute(0, 2, 1).contiguous()  # [C, 312*12, 64]
#         cam_features = cam_features.view(-1, self.cam_channel)  # [C*312*12, 64]
        
#         cam_encoded = self.cam_encoder(cam_features)  # [C*312*12, 64]
#         cam_encoded = cam_encoded.view(num_cameras, self.cam_height*self.cam_width, -1)  # [C, 312*12, 64]
#         cam_agg = cam_encoded.mean(dim=1).mean(dim=0)  # [64] (공간+카메라 집계)
        
#         # 라이다 특징 추출
#         lidar_encoded = self.lidar_encoder(detection_xyz_normal)  # [N, 64]
        
#         # 교차 모달리티 융합
#         fused = torch.cat([
#             lidar_encoded,
#             cam_agg.unsqueeze(0).expand(num_objects, -1)
#         ], dim=1)  # [N, 128]
        
#         # 잔차 연결 기반 좌표 보정
#         return detection_xyz_normal[...,2:] + self.fusion(fused)

# class DifferentiableQueryProcessor(torch.nn.Module):
#     def __init__(self, num_cameras=6, max_objects=100, num_points=100):
#         super().__init__()
#         self.num_points = num_points
#         self.max_objects = max_objects
#         self.num_cameras = num_cameras
#         self.obj_emb = torch.nn.Embedding(max_objects, 16)
#         self.cam_emb = torch.nn.Embedding(num_cameras, 16)
        
#     def forward(self, corrs, sbs_img):
#         device = corrs.device
        
#         # 1. 카메라별 마스크 생성
#         cam_masks = F.one_hot(corrs[:,0].long(), self.num_cameras).bool()
        
#         # 2. 카메라별 특징 분리
#         all_cam_features = []
#         for cam_idx in range(self.num_cameras):
#             cam_mask = cam_masks[:, cam_idx]
#             cam_corrs = corrs[cam_mask]  # [M, 9]
            
#             if cam_corrs.size(0) == 0:
#                 padded = torch.zeros((self.num_points, 9), device=device)
#                 all_cam_features.append(padded)
#                 continue
                
#             # 3. 객체 처리 (전체 9개 특징 사용)
#             obj_ids = torch.clamp(cam_corrs[:,1].long(), 0, self.max_objects-1)
#             obj_features = self.obj_emb(obj_ids)
            
#             obj_logits = obj_features @ self.obj_emb.weight.T
#             obj_counts = torch.sum(F.softmax(obj_logits, dim=-1), dim=0)
#             sorted_idx = torch.argsort(obj_counts, descending=True)
            
#             samples = F.gumbel_softmax(obj_logits[:, sorted_idx], tau=0.5, hard=True)
#             selected = torch.matmul(samples.T, cam_corrs)  # ← 핵심 수정 부분 ([:, 2:] → [:] )
            
#             # 4. 동적 패딩
#             if selected.size(0) < self.num_points:
#                 padding = torch.zeros((self.num_points-selected.size(0), 9), device=device)
#                 trimmed = torch.cat([selected, padding], dim=0)
#             else:
#                 trimmed = selected[:self.num_points]
                
#             all_cam_features.append(trimmed)

#         # 5. 최종 출력 형식
#         trimmed_corrs = torch.stack(all_cam_features, dim=0)  # [6, 100, 9]
        
#         # 6. 이미지 선택
#         cam_weights = torch.sigmoid(torch.sum(cam_masks.float(), dim=0) * 1e3)
#         selected_imgs = torch.einsum('c,c...->c...', cam_weights, sbs_img)
        
#         return selected_imgs, trimmed_corrs, torch.unique(corrs[:,0].long())

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


class ZEstimator(nn.Module):
    def __init__(self, enc_channels=312, bbox_channels=256, uv_dim=2, hidden_dim=512):
        super().__init__()
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


@HEADS.register_module()
class MV2DSHead(MV2DHead):
    def __init__(self,
                 # denoise setting
                 use_denoise=False,
                 neg_bbox_loss=False,
                 denoise_scalar=10,
                 denoise_noise_scale=1.0,
                 denoise_noise_trans=0.0,
                 denoise_weight=1.0,
                 denoise_split=0.75,
                 **kwargs):
        super(MV2DSHead, self).__init__(**kwargs)
        self.use_denoise = use_denoise
        self.neg_bbox_loss = neg_bbox_loss
        self.denoise_scalar = denoise_scalar
        self.denoise_noise_scale = denoise_noise_scale
        self.denoise_noise_trans = denoise_noise_trans
        self.denoise_weight = denoise_weight
        self.denoise_split = denoise_split
        
        # self.corr_loss = CorrelationCycleLoss(corr_weight=1.0 , cycle_weight=0.5)
        # self.point_distance_loss = PointDistanceLoss(distance_weight=1.0)
        
        self.num_kp =200
        # self.conf_loss_weight = 0.5
        self.corr = COTR(self.num_kp) 
        self.fine_corr = GraphBEVLocalAlignNet() 
        # self.corr_loss = SelfSupervisedCorrespondenceLoss(
        #                     cycle_weight=1.0,
        #                     photo_weight=0.3,
        #                     geom_weight=0.2,
        #                     smooth_weight=0.1,
        #                     range_weight=0.05
        #                 )
       
        # self.z_estimator = ZValueEstimator(depth_shape=(900, 1600))
        # self.z_estimator = BBoxEnhancedZEstimator(depth_shape=(900, 1600))  # 예시로 depth_shape 설정
        # self.z_estimator = ImprovedDepthEstimator()
        # self.z_estimator = SimplifiedDepthEstimator()
        self.z_estimator = ZEstimator(enc_channels=312, bbox_channels=256, uv_dim=2, hidden_dim=512)
        # self.pts_regressor = pts_regressor()
        # self.query_selector = DifferentiableQueryProcessor(num_cameras=6,max_objects=100, num_points=100)

        # # Deformable SPN 레이어 추가
        # self.deform_spn = DeformableSPN()
    
    def create_confidence_cross_attention_mask(self, confidence_scores, confidence_threshold=0.3):
        """
        Confidence 기반 cross-attention mask 생성
        Args:
            confidence_scores: [N] tensor, confidence values (0~1)
            confidence_threshold: float, confidence threshold
        Returns:
            cross_attn_mask: [N, num_corrs, H, W] boolean mask
        """
        # High confidence mask (True = attend, False = mask out)
        high_confidence_mask = confidence_scores > confidence_threshold
        
        return high_confidence_mask

    def apply_confidence_to_corr_feats(self, corr_feats, confidence_scores, confidence_threshold=0.3):
        """
        Confidence 기반으로 correlation features에 가중치 적용
        Args:
            corr_feats: [num_rois, num_corrs, c, h, w]
            confidence_scores: [num_rois] 
        Returns:
            weighted_corr_feats: confidence로 가중치가 적용된 features
        """
        N, num_corrs, c, h, w = corr_feats.shape
        
        # Confidence를 feature dimension에 맞게 확장
        conf_weights = confidence_scores.view(N, 1, 1, 1, 1).expand(N, num_corrs, c, h, w)
        
        # Confidence threshold 적용
        conf_weights = torch.where(
            confidence_scores.view(N, 1, 1, 1, 1).expand(N, num_corrs, c, h, w) > confidence_threshold,
            conf_weights,
            torch.zeros_like(conf_weights)
        )
        
        # Feature에 confidence 가중치 적용
        weighted_corr_feats = corr_feats * conf_weights
        
        return weighted_corr_feats

    def prepare_for_dn(self, batch_size, reference_points, img_metas, ref_num, eps=1e-4):
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d'].gravity_center, img_meta['gt_bboxes_3d'].tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d'] for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.denoise_scalar, 1).view(-1)
            known_labels = labels.repeat(self.denoise_scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.denoise_scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.denoise_scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.denoise_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.denoise_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.denoise_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0 + eps, max=1.0 - eps)
                mask = torch.norm(rand_prob, 2, 1) > self.denoise_split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.denoise_scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat(
                    [map_known_indice + single_pad * i for i in range(self.denoise_scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + ref_num
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.denoise_scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.denoise_scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict
    
    def inverse_layer_norm(self, normalized, ln_layer):
        return (normalized - ln_layer.bias) / ln_layer.weight
    
    # 출력 범위 강제 조정
    # def constrain_output(self, x, min_val, max_val):
    #     """Sigmoid + Scaling으로 출력 범위 제한"""
    #     sig = torch.sigmoid(x)  # [0,1] 범위로 압축
    #     return min_val + (max_val - min_val) * sig
    
    def constrain_output (self,x, min_val, max_val):
        """Tanh 기반으로 gradient 소실 방지"""
        tanh_out = torch.tanh(x)  # [-1, 1] 범위
        scaled = (tanh_out + 1) * 0.5  # [0, 1] 범위로 변환
        return min_val + (max_val - min_val) * scaled

    def _bbox_forward_denoise(self, img,img_metas,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): # for SJMOON
    # def _bbox_forward_denoise(self, x, proposal_list, img_metas): # for original 
        # avoid empty 2D detection
        if sum([len(p) for p in proposal_list]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposal_list[0].dtype,
                                    device=proposal_list[0].device)
            proposal_list = [proposal] + proposal_list[1:]

        rois = bbox2roi(proposal_list)
        uv_set = gt_KT_3by4 # uv value setting (u,v,u',v')
        object_indices = torch.arange(start=0,end=rois.size(0),dtype=torch.int32,device=rois.device).unsqueeze(1)
        rois_part1 = rois[:, :1]   # 이미지 인덱스 [3,1]
        rois_part2 = rois[:, 1:]    # 좌표 정보 [3,4]
        rois_with_indices = torch.cat([rois_part1, object_indices, rois_part2], dim=1)
        rois_center = get_center_points(rois_with_indices)

        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        # bbox_feats, obj_ids = self.idawareroiextractor(x[:self.bbox_roi_extractor.num_inputs], rois_with_indices)

        # 3dpe was concatenated to fpn feature
        c = bbox_feats.size(1)
        bbox_feats, pe = bbox_feats.split([c // 2, c // 2], dim=1)

        # intrinsics as extra input feature
        extra_feats = dict(
            intrinsic=self.process_intrins_feat(rois, intrinsics)
        )

        ###### SJ MOON 수정 #############
        # with torch.no_grad():
        dense_depth_map_gt = dense_map_from_depth_batch(uvz_gt.squeeze(0),grid=3,iterations=3)
        dense_depth_map = dense_map_from_depth_batch(lidar_depth_mis,grid=3,iterations=3)
        dense_depth_img_mis = dense_depth_map.to(dtype=torch.uint8)
        dense_depth_img_color_mis = batch_colormap(dense_depth_img_mis)
        # dense_depth_img_color_mis = differentiable_colormap(dense_depth_img_mis)
        
        img_resized = F.interpolate(img, size=[192, 640], mode="bilinear")
        lidar_depth_mis_resized = F.interpolate(dense_depth_img_color_mis, size=[192, 640], mode="bilinear")
        lidar_depth_mis_resized = tvtf.normalize(lidar_depth_mis_resized, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
       
        # lidar_depth_mis_resized = F.interpolate(lidar_depth_mis, size=[h, w], mode="bilinear")
        
        # Deformable SPN 적용 (주요 수정 부분)
        # dense_depth = self.deform_spn(lidar_depth_mis_resized)
        # dense_depth = geometric_propagation(lidar_depth_mis_resized)

        sbs_img = two_images_side_by_side(img_resized, lidar_depth_mis_resized)
        sbs_img = torch.from_numpy(sbs_img).permute(0,3,1,2)
       
        sbs_img = two_images_side_by_side_gpu(img_resized, lidar_depth_mis_resized)
        sbs_img = sbs_img.permute(0,3,1,2)
       
        # sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # ############## input display ##########################
        # visualize_bboxes(img,proposal_list,output_path='bbox_display.png')
        
        # display_depth_maps(img,dense_depth_img_color_mis,sbs_img,uv_set)
        # print("input dispaly end")

        # ### query generator
        # ref_points_uvz,reference_points,return_feats = self.query_generator(bbox_feats, intrinsics, extrinsics, extra_feats)
        # reference_points_with_indices = torch.cat([rois_with_indices[:,0:2],ref_points_uvz],dim=1)
        # reference_points_raw = reference_points.clone()
        # reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (
        #         self.pc_range[3] - self.pc_range[0])
        # reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (
        #         self.pc_range[4] - self.pc_range[1])
        # reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (
        #         self.pc_range[5] - self.pc_range[2])
        # reference_points = reference_points.clamp(min=0, max=1)
        # reference_points_with_indices= torch.cat([rois_with_indices[:,0:2],reference_points],dim=1)
        # reference_points_with_indices1= torch.cat([rois_with_indices[:,0:1],reference_points_raw],dim=1)
        # points_lidar2img, mask_valid = lidar_to_image_with_index(reference_points_with_indices1,gt_KT,img_shape=(928,1600))
        
        # # ########### corr transformer sjmoon ###########
        trimed_center_pts =batch_rois_center_by_cam_id(rois_center,batch_size=200)
        trimed_uvset = batched_trim_corrs(uv_set).to(dtype=torch.float32, device=img.device)
        # 객체 ID 보존 텐서
        object_ids = trimed_center_pts[..., 1].clone()  # [num_cams, batch_size]

        # 쿼리 입력 생성 (좌표만 정규화)
        # query_input = trimed_uvset[..., :2]
        query_input = trimed_center_pts[..., 2:].clone()  # [num_cams, batch_size, 2]
        # scaled1_query_input = scale_uvz_points(query_input,original_size=(900,1600),target_size=(192,640))
        # scaled2_query_input = normalize_uv_points(scaled1_query_input)

        query_input[..., 0] /= 1600.0
        query_input[..., 1] /= 928.0

        corr_target = trimed_uvset[...,2:]
        corr_target[...,0] = corr_target[...,0] / 1600.0
        corr_target[...,1] = corr_target[...,1] / 920.0

        query_input[:,:,0] = query_input[:,:,0]/2    # recaling points for sbs image resizing
        query_input[:,:,1] = query_input[:,:,1]
        corr_target[:,:,0] = corr_target[:,:,0]/2 + 0.5 # recaling points for sbs image resizing
        corr_target[:,:,1] = corr_target[:,:,1] 

        raw_corrs, cycle, corr_mask, enc_out = self.corr(sbs_img, query_input)
        # 객체 ID 정보를 예측 결과에 연결

        
        # loss_corr = self.corr_loss(raw_corrs, corr_target, cycle, query_input, corr_mask)
        fine_raw_corrs = self.fine_corr(raw_corrs, dense_depth_map)
        # fine_raw_corrs[...,0] = fine_raw_corrs[...,0] - 0.5
        # loss_corr = self.corr_loss(fine_raw_corrs[...,:2], query_input, img, dense_depth_map)
        
        corrs_pred_with_obj = torch.cat([
            object_ids.unsqueeze(-1),  # [num_cams, batch_size, 1]
            fine_raw_corrs                 # [num_cams, batch_size, 2]
        ], dim=-1)  # [num_cams, batch_size, 3
        # corr_loss = self.corr_loss(corrs_pred, corr_target, cycle, query_input, corr_mask)
        raw_pred_center_pts = remove_duplicate_objs(corrs_pred_with_obj)
        # pred_center_pts1 = denormalize_uv_points(raw_pred_center_pts)
        # # pred_center_pts1[...,2] -= 640  # x 좌표 보정
        # pred_center_pts = descale_uvz_points(pred_center_pts1,original_size=(192,640),target_size=(900,1600))
        # restore_original_uv

        raw_pred_center_pts1 = raw_pred_center_pts.clone()
        raw_pred_center_pts1[..., 2] = (raw_pred_center_pts1[..., 2] - 0.5) * 2
        # raw_pred_center_pts1[..., 3] = raw_pred_center_pts1[..., 3] * 2
        raw_pred_center_pts2 = raw_pred_center_pts1.clone()
        raw_pred_center_pts2[..., 2] *= 1600.0
        raw_pred_center_pts2[..., 3] *= 928.0

        # # ##### 검증용 display ######
        # from image_processing_unit_Ver15_0 import draw_correspondences
        # # corrs_pred_norm = self.inverse_layer_norm(corrs_pred, self.final_ln)
        # # rois_center_disp = scale_uvz_points(rois_center[...,2:],original_size=(900,1600),target_size=(192,640))
        # # trimed_corrs = batch_rois_center_by_cam_id(rois_center,batch_size=200)
        # # pred_corrs = torch.cat([rois_center_disp,pred_center_pts1[...,2:]],dim=-1)
        # gt_corrs = torch.cat([query_input,corr_target],dim=-1)
        # pred_corrs = torch.cat([query_input,raw_corrs],dim=-1)
        # # int_ids = original_camera_ids.to(torch.long).cpu()
        # # if len(int_ids) < 6:
        # #     print ("len(int_ids) < 6")
        # # 카메라 ID ↔ 인덱스 매핑 생성
        # # id_to_idx = {cid.item(): idx for idx, cid in enumerate(original_camera_ids)}
        # # for cid in int_ids :
        # for cid in range(6):
        #     # idx = id_to_idx[cid.item()]
        #     draw_correspondences(
        #         trimed_corrs = gt_corrs[cid],  # 첫 번째 배치 선택
        #         sbs_img=sbs_img[cid],
        #         save_path='correspondence_visualization_gt.jpg'
        #     )
        #     draw_correspondences(
        #         trimed_corrs = pred_corrs[cid][:2,...],  # 첫 번째 배치 선택
        #         sbs_img=sbs_img[cid],
        #         save_path='correspondence_visualization_pred.jpg'
        #     )
        #     print ("end")

        # transformed_uv = transform_uv_points(rois_with_indices,uv_set)      
        # esitmated_z = self.z_estimator(transformed_uv[...,:4], dense_depth_map_gt,bbox_feats,ref_points_uvz)
        # esitmated_z = self.z_estimator(pred_center_pts, dense_depth_map_gt,bbox_feats)
        esitmated_z = self.z_estimator(raw_pred_center_pts2, dense_depth_map,bbox_feats, enc_out)
        # **Confidence 정보 추출**
        confidence_scores = esitmated_z['confidence'].view(-1,1)  # [N]
        # z_depth_real = esitmated_z['z_lidar_real']  # [N]
        fine_z_raw = raw_pred_center_pts2[..., 4].reshape(-1, 1)  # [N, 1]
        # 융합된 z 계산 (예: confidence 가중 평균)
        z_fused = confidence_scores * fine_z_raw + (1 - confidence_scores) * esitmated_z['depth']  # [N, 1]
        esitmated_uvz =torch.cat([raw_pred_center_pts2[...,:4], z_fused],dim=1)

        # # Confidence 손실 계산
        # conf_loss = F.binary_cross_entropy(
        #     esitmated_z['confidence'], 
        #     (z_depth_real > 0.1).float(),  # 유효한 depth가 있는 경우 1, 없는 경우 0
        #     reduction='mean'
        # )

        detection_nonzero_xyz = image_to_lidar_global_modi3(esitmated_uvz,gt_KT) # 교정되어진 lidar좌표계 pc
        
        # scaled_transformed_uv = scale_uvz_points(transformed_uv[...,2:4],original_size=(928,1600),target_size=(192,640))
        # scaled_transformed_uv_prime = scale_uvz_points(transformed_uv[...,6:],original_size=(900,1600),target_size=(192,640))
        # scaled_transformed_uv_set = torch.cat([transformed_uv[...,:2],scaled_transformed_uv,scaled_transformed_uv_prime],dim=1)

        # detection_nonzero_uvz_with_ObjectID =find_rois_nonzero_z_adv8(rois_with_indices,dense_depth_map_gt.unsqueeze(0))
        # # detection_nonzero_uvz_with_ObjectID =differentiable_find_rois1(rois_with_indices,uvz_gt)
        # detection_nonzero_xyz = image_to_lidar_global_modi3(esitmated_uvz,gt_KT) # 교정되어진 lidar좌표계 pc
        # detection_real_mask = (detection_nonzero_uvz_with_ObjectID[:,5] == 1.0) \
        #                     | (detection_nonzero_uvz_with_ObjectID[:,5] == 0.7) \
        #                     | (detection_nonzero_uvz_with_ObjectID[:,5] == 0.3) \
        #                     # | (detection_nonzero_uvz_with_ObjectID[:,5] == 0.3) \
        #                     # | (detection_nonzero_uvz_with_ObjectID[:,5] == 0.1) \
        # # detection_pred_mask = (detection_nonzero_uvz_with_ObjectID[:,5] == 0.7)
        # detection_uvz_lidar = detection_nonzero_uvz_with_ObjectID[detection_real_mask]
        # detection_xyz_lidar = detection_nonzero_xyz[detection_real_mask]
        
        # assert detection_xyz_lidar.shape[0] == rois_with_indices.shape[0], "detection_uvz_lidar and reference_points_with_indices shape mismatch"

        # # merge_xyz = merge_point_clouds(reference_points_with_indices, detection_xyz_lidar[...,:5])
        
        # ########### corr 예측을 위한 preprocessing ###########
        # gt_xyz = miscalib_transform2(detection_xyz_lidar,mis_Rt)
        # gt_xyz_lidar = gt_xyz[detection_real_mask]

        detection_xyz = detection_nonzero_xyz[:,2:5].clone()
        # pts_lidar_mis = gt_xyz_lidar[:,2:5].clone()
        detection_xyz_normalized = detection_xyz.clone()
        # pts_lidar_mis_normalized = pts_lidar_mis.clone()

        detection_xyz_normalized[..., 0:1] = (detection_xyz_normalized[..., 0:1] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0])
        detection_xyz_normalized[..., 1:2] = (detection_xyz_normalized[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        detection_xyz_normalized[..., 2:3] = (detection_xyz_normalized[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        detection_xyz_normalized = detection_xyz_normalized.clamp(min=0, max=1)

        # pts_lidar_mis_normalized[..., 0:1] = (pts_lidar_mis_normalized[..., 0:1] - self.pc_range[0]) / (
        #     self.pc_range[3] - self.pc_range[0])
        # pts_lidar_mis_normalized[..., 1:2] = (pts_lidar_mis_normalized[..., 1:2] - self.pc_range[1]) / (
        #         self.pc_range[4] - self.pc_range[1])
        # pts_lidar_mis_normalized[..., 2:3] = (pts_lidar_mis_normalized[..., 2:3] - self.pc_range[2]) / (
        #         self.pc_range[5] - self.pc_range[2])
        # pts_lidar_mis_normalized = pts_lidar_mis_normalized.clamp(min=0, max=1)
        # pts_lidar_mis_normalized_with_index = torch.cat([gt_xyz_lidar[...,0:2],pts_lidar_mis_normalized],dim=1)
        
        # det_xyz_hom_with_index = torch.cat([detection_xyz_lidar[:,0:1],detection_xyz],dim=1)
        # # pts_hom = torch.cat([pts_lidar_mis, torch.ones_like(pts_lidar_mis[:, :1])], dim=1)
        # pts_hom_with_index = torch.cat([gt_xyz_lidar[:,0:1],pts_lidar_mis],dim=1)
        
        # points_lidar2img, mask_valid = lidar_to_image_with_index(det_xyz_hom_with_index,gt_KT,img_shape=(900,1600))
        # points_lidar2img_mis ,mask_valid_mis = lidar_to_image_with_index(pts_hom_with_index,gt_KT,img_shape=(900,1600))
        # common_mask = mask_valid & mask_valid_mis 
        # # common_mask= (mask_valid > 0.5) & (mask_valid_mis > 0.5)

        # points_lidar2img = points_lidar2img[common_mask]
        # # points_lidar2img_mis = points_lidar2img_mis[common_mask]

        # assert points_lidar2img.shape[0] == points_lidar2img_mis.shape[0], "points_lidar2img and points_lidar2img_mis shape mismatch"

        # scaled_points_lidar2img = scale_uvz_points(points_lidar2img[:,1:],original_size=(928,1600),target_size=(192,640))
        # scaled_points_lidar2img_mis = scale_uvz_points(points_lidar2img_mis[:,1:],original_size=(900,1600),target_size=(192,640))
        
        # normal_points_lidar2img_mis = normalize_uvz_points(scaled_points_lidar2img_mis)
        # normal_points_lidar2img_mis[:, 0] += 0.5
        # normal_points_lidar2img = normalize_uvz_points(scaled_points_lidar2img)
        # # corrs_points = torch.cat([detection_xyz_lidar[mask_valid_mis][:,0:2],normal_points_lidar2img,normal_points_lidar2img_mis],dim=1)
        # corrs_points = torch.cat([detection_xyz_lidar[common_mask][:,0:2],
        #                         normal_points_lidar2img,
        #                         normal_points_lidar2img_mis,
        #                         pts_lidar_mis_normalized_with_index[common_mask][:,2:5],
        #                         detection_xyz_lidar[common_mask][:,5:6]],
        #                         dim=1)
    
        # selected_imgs, trimed_corrs ,original_camera_ids = process_queries_adv(corrs_points,sbs_img)
        # selected_imgs, trimed_corrs ,original_camera_ids = process_queries_adv_modified(scaled_transformed_uv_set,sbs_img)
        # # selected_imgs, trimed_corrs ,original_camera_ids = differentiable_process_queries(corrs_points.float(),sbs_img)

        # # ########### corr transformer sjmoon ###########
        # query_input = trimed_corrs[...,:2]
        # corr_target = trimed_corrs[...,2:]
        # # pts_target = trimed_corrs[...,8:11]
    
        # corrs_pred, cycle, corr_mask, enc_out = self.corr(selected_imgs, query_input)
        # corr_loss = self.corr_loss(corrs_pred, corr_target, cycle, query_input, corr_mask)
       
        # corrs_pred_with_index = torch.cat([trimed_corrs[...,0:2],corrs_pred],dim=2)
        # deduplicated_corrs_pred = deduplicate_obj_ids(corrs_pred_with_index)
        # denormal_corrs_pred = denormalize_points(deduplicated_corrs_pred[...,2:])
        # descale_corrs_pred = inverse_scale_uvz_points(denormal_corrs_pred,original_size=(928, 1600), target_size=(192, 640))
        # descale_corrs_pred_with_indices = torch.cat([deduplicated_corrs_pred[:,0:2],descale_corrs_pred],dim=1)
        # bbox_pred_uvz = convert_to_bbox_coordinates_matched(rois_with_indices,descale_corrs_pred_with_indices,self.roi_size)

        # pts_x = self.pts_regressor(corrs_pred,enc_out)
        # pc_distance_loss = self.point_distance_loss(pts_x,pts_target)

        # pts_x_with_index = torch.cat([trimed_corrs[...,0:2],pts_x],dim=2)
        # filtered_tensor = differentiable_deduplicate(pts_x_with_index, temperature=0.1)
        # filterd_pts_x = filtered_tensor.clone().float()
        
        # skip_detection_xyz_normal = self.skip_connection(detection_xyz_normal[...,2:],enc_out)
       
        # with torch.no_grad():  # 객체 ID는 미분 흐름에서 제외

        #############
        # denormal_pred_uvz = denormalize_points(corrs_pred)
        # descale_pre_uvz = inverse_scale_uvz_points(denormal_pred_uvz,original_size=(928, 1600), target_size=(192, 640))
        # descale_pre_uvz_with_index = torch.cat([trimed_corrs[...,0:2],descale_pre_uvz],dim=2)
        # detection_xyz_adv_with_index = image_to_lidar_global_modi2(descale_pre_uvz_with_index,gt_KT,original_camera_ids) # 교정되어진 lidar좌표계 pc
        ##########
        
        # pixel_normal_uvz = pixel_to_normalized(descale_pre_uvz_with_index,intrinsics)
        # detection_xyz_adv_with_index, detection_xyz_adv ,lidar2img = center2lidar_batch(pixel_normal_uvz,intrinsics,extrinsics)
        # filtered_tensor = deduplicate_obj_ids(detection_xyz_adv_with_index)
        
        #     # # # 객체 ID 기반 정렬 및 필터링
        # det_obj_ids = detection_xyz_normal[:,1].int()
        # pts_obj_ids = pts_lidar_mis_normalized_with_index[:,1].int()
        # det_unique = torch.unique(det_obj_ids)
        # pts_unique = torch.unique(pts_obj_ids)
        # common_ids = det_unique[torch.isin(det_unique, pts_unique)]
        # # 공통 ID에 대한 마스크 생성
        # det_mask = torch.isin(det_obj_ids, common_ids)
        # pts_mask = torch.isin(pts_obj_ids, common_ids)
        # pred_pts = detection_xyz_normal[det_mask]
        # gt_pts = pts_lidar_mis_normalized_with_index[pts_mask]
        
        # detection_xyz_adv_with_index, detection_xyz_adv ,lidar2img = differentiable_center2lidar(pixel_normal_uvz,intrinsics,extrinsics)
        
        #######
        # filtered_tensor = differentiable_deduplicate(detection_xyz_adv_with_index, temperature=0.1)
        # detection_xyz_normal = filtered_tensor.clone().float()
        ##########
        
        # detection_xyz_normal = torch.where(
        #     torch.isnan(detection_xyz_normal),
        #     torch.zeros_like(detection_xyz_normal),
        #     detection_xyz_normal
        # )

        # ## query generator by SJMOON : 카메라에서 제대로 나온 라이다 xyz points 
        # detection_xyz_normal = torch.cat([
        #     detection_xyz_normal[..., :2],
        #     (detection_xyz_normal[..., 2:3] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]),
        #     (detection_xyz_normal[..., 3:4] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]),
        #     (detection_xyz_normal[..., 4:5] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        # ], dim=-1)

        # detection_xyz_normal = torch.cat([
        #                                     detection_xyz_normal[..., :2],
        #                                     detection_xyz_normal[..., 2:].clamp(min=0, max=1)
        #                                 ], dim=-1)

        # pred_pts, gt_pts = differentiable_object_matching(
        #                 detection_xyz_normal, 
        #                 pts_lidar_mis_normalized_with_index,
        #                 # obj_id_dim=1,
        #                 # temp=0.1
        #                 )
        
        # pc_distance_loss = self.point_distance_loss(pred_pts[...,2:],gt_pts[...,2:])

        # ref_points_with_index = merge_point_clouds(reference_points_with_indices, detection_xyz_normal)
        # ref_points_with_index = differentiable_merge_point_clouds(reference_points_with_indices, gt_pts)
        ref_points = detection_xyz_normalized
         
        # generate box correlation
        corr, mask = self.box_corr_module.gen_box_roi_correlation(rois, [len(p) for p in proposal_list], img_metas)
        # corr, mask = self.box_corr_module.gen_box_roi_correlation(rois_with_indices,pred_pts, [len(p) for p in proposal_list], img_metas)

        # depth_x_feats, dapth_pe = depth_x[0].split([ 512 // 2, 512 // 2], dim=1)

        # B, C, H, W = depth_x_feats.shape
        # current_tgt_len = corrs_pred_normalization.shape[0]  # 63
        # conf_mask = conf_scores.view(current_tgt_len, 1, 1, 1)  # [63, 1, 1, 1]

        # # [n_query, n, h, w] 형태로 expand
        # conf_mask = conf_mask.expand(-1, B, H, W)  # [63, 6, 32, 88]

        # # auto-calib cross attention module 
        # pred_xyz_feat = self.bbox_head.forward_calib_attn(detection_xyz_normal[None,:,:],
        #                                 depth_x_feats[None], # x
        #                                 torch.zeros_like(depth_x_feats[None, :, 0]).bool(), #masks
        #                                 dapth_pe[None], # x position embedding 
        #                                 attn_mask=None,
        #                                 cross_attn_mask=None,
        #                                 force_fp32=self.force_fp32)
                
        # batch_size, seq_len, num_points, feature_dim = pred_xyz_feat.shape
        # pred_xyz_feat = pred_xyz_feat.view(num_points,batch_size*seq_len*feature_dim)
        # autocal_pred_xyz = self.dynamic_linear(pred_xyz_feat)

        # loss_corr = self.corr_loss(autocal_pred_xyz, pts_lidar_mis_normal) # arguments : corr_pred, corr_target, cycle, queries, mask
        # pred_xyz = torch.cat([autocal_pred_xyz, detection_xyz_pred_normal],dim=0)

        # ####### input 검증용 #############
        # int_ids = original_camera_ids.to(torch.long).cpu()
        # if len(int_ids) < 6:
        #     print ("len(int_ids) < 6")
        # # 카메라 ID ↔ 인덱스 매핑 생성
        # id_to_idx = {cid.item(): idx for idx, cid in enumerate(original_camera_ids)}
        # for cid in int_ids :
        #     i = id_to_idx[cid.item()]
        #     ref_lidar_img = detection_xyz.matmul(gt_KT[i][:3, :3].T) + gt_KT[i][:3, 3].unsqueeze(0)
        #     ref_lidar_img = torch.cat([ref_lidar_img[:, :2] / ref_lidar_img[:, 2:3], ref_lidar_img[:, 2:3]], 1)
        #     depth , ref_uv,ref_z, valid_indices = points2depthmap(ref_lidar_img , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     ori_uvz = torch.cat((ref_uv, ref_z.unsqueeze(1)), dim=1)
        #     dense_depth_img_raw = dense_map_gpu_optimized(ori_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     dense_depth_img_raw = dense_depth_img_raw.to(dtype=torch.uint8).to(detection_xyz.device)
        #     dense_depth_img_color_raw = colormap(dense_depth_img_raw)

        #     ######### mis-aligned ##########
        #     # 1. Homogeneous 좌표로 변환
        #     pts_hom = torch.cat([pts_lidar_mis, torch.ones_like(pts_lidar_mis[:, :1])], dim=1)
        #     points_img_mis = (gt_KT[i] @ pts_hom.T).T
        #     points_img_mis = torch.cat([points_img_mis[:, :2] / points_img_mis[:, 2:3], points_img_mis[:, 2:3]], 1)
        #     depth ,comp_uv,comp_z, valid_indices = points2depthmap(points_img_mis , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     comp_uvz = torch.cat((comp_uv, comp_z.unsqueeze(1)), dim=1)
        #     dense_depth_img_mi_comp = dense_map_gpu_optimized(comp_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     dense_depth_img_mi_comp = dense_depth_img_mi_comp.to(dtype=torch.uint8).to(detection_xyz.device)
        #     dense_depth_img_color_mis_comp = colormap(dense_depth_img_mi_comp)

        #     #### 예측값 디스플레이 ####
        #     denormalized_pts = ref_points.clone()
        #     denormalized_pts[..., 0:1] = ref_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        #     denormalized_pts[..., 1:2] = ref_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        #     denormalized_pts[..., 2:3] = ref_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        #     lidar_points_pred_homo = torch.cat([denormalized_pts, torch.ones_like(denormalized_pts[:, :1])], dim=1)
        #     points_img_pred = (gt_KT[i] @ lidar_points_pred_homo.T).T
        #     points_img_pred = torch.cat([points_img_pred[:, :2] / points_img_pred[:, 2:3], points_img_pred[:, 2:3]], 1)
        #     depth , pred_uv,pred_z, valid_indices = points2depthmap(points_img_pred , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     pred_uvz = torch.cat((pred_uv, pred_z.unsqueeze(1)), dim=1)
        #     dense_depth_img_mis = dense_map_gpu_optimized(pred_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     dense_depth_img_mis = dense_depth_img_mis.to(dtype=torch.uint8).to(detection_xyz.device)
        #     dense_depth_img_color_mis = colormap(dense_depth_img_mis)

        #     ###### 검증용 display########
        #     import matplotlib.pyplot as plt
        #     img_np = img[i].permute(1, 2, 0).detach().cpu().numpy()
        #     # lidar_depth_mis_np = lidar_depth_mis[0].permute(1, 2, 0).detach().cpu().numpy()
        #     if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        #         img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
        #     ref_depth_np = dense_depth_img_color_raw.detach().cpu().numpy()
        #     comp_mis_depth_np = dense_depth_img_color_mis_comp.detach().cpu().numpy()
        #     mis_depth_np = dense_depth_img_color_mis.detach().cpu().numpy()
        #     # depth_np = dense_depth_img_color_mis_gt.detach().cpu().numpy()

        #     # 깊이 맵의 알파 채널 설정 (투명도 조절)
        #     alpha = 0.5
        #     ref_depth_np_with_alpha = np.concatenate([ref_depth_np, np.ones((*ref_depth_np.shape[:2], 1)) * alpha], axis=2)
        #     # depth_np_with_alpha = np.concatenate([depth_np, np.ones((*depth_np.shape[:2], 1)) * alpha], axis=2)
        #     mis_depth_np_with_alpha = np.concatenate([mis_depth_np, np.ones((*mis_depth_np.shape[:2], 1)) * alpha], axis=2)
        #     comp_mis_depth_np_with_alpha = np.concatenate([comp_mis_depth_np, np.ones((*comp_mis_depth_np.shape[:2], 1)) * alpha], axis=2)

        #     # 그림 생성
        #     fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(40, 20))

        #     ## 첫 번째 서브플롯: img와 depth_np 오버레이
        #     ax1.imshow(img_np)
        #     # ax1.imshow(depth_np_with_alpha)
        #     ax1.imshow(ref_depth_np_with_alpha)
        #     ax1.set_title("original_ref", fontsize=10)
        #     ax1.axis('off')

        #     # # 두 번째 서브플롯: img와 mis_depth_np 오버레이
        #     ax2.imshow(img_np)
        #     ax2.imshow(comp_mis_depth_np_with_alpha)
        #     ax2.set_title("Object depth Map", fontsize=10)
        #     ax2.axis('off')

        #     # # 세 번째 서브플롯: img와 mis_depth_np 오버레이
        #     ax3.imshow(img_np)
        #     ax3.imshow(mis_depth_np_with_alpha)
        #     ax3.set_title("Object mis prediction depth Map", fontsize=10)
        #     ax3.axis('off')

        #     # 전체 그림 저장
        #     plt.tight_layout()
        #     plt.savefig('verify.jpg', dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print ("end")

        if self.use_denoise and self.training:
            # bbox_feats: [num_rois, c, h, w]
            n_rois, c, h, w = bbox_feats.shape
            cross_attn_mask = bbox_feats.new_ones((n_rois, n_rois + 1)).bool()
            corr[~mask] = n_rois  # [num_rois, max_corr]
            cross_attn_mask = torch.scatter(cross_attn_mask, 1, corr, 0)
            cross_attn_mask = cross_attn_mask[:, :n_rois, None, None].expand(n_rois, n_rois, h, w)

            reference_points_ori = reference_points

            reference_points, attn_mask, mask_dict = self.prepare_for_dn(1, reference_points, img_metas[0:1],
                                                                         len(reference_points))
            reference_points = reference_points[0]

            cross_attn_mask_pad = cross_attn_mask.new_zeros(
                (len(reference_points) - len(reference_points_ori), n_rois, h, w))

            cross_attn_mask = torch.cat([cross_attn_mask_pad, cross_attn_mask])

            all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[None],
                                                            bbox_feats[None],
                                                            torch.zeros_like(bbox_feats[None, :, 0]).bool(),
                                                            pe[None],
                                                            attn_mask=attn_mask,
                                                            cross_attn_mask=cross_attn_mask,
                                                            force_fp32=self.force_fp32, )
        else:
            mask_dict = None

            corr_feats = bbox_feats[corr]  # [num_rois, num_corrs, c, h, w]
            corr_pe = pe[corr]

            ##### 가변 reference point by SJMOON ######
            # output_size = corr_feats.shape[0] * 3
            # # dynamic_linear = nn.Linear(600 * 3, output_size).to(corr_feats.device)
            # # reference_points_modified = dynamic_linear(reference_points_modified)

            # # **Confidence 기반 Feature Weighting 적용**
            # confidence_threshold = 0.5  # 임계값 설정
            # weighted_corr_feats = self.apply_confidence_to_corr_feats(
            #     corr_feats, confidence_scores, confidence_threshold
            # )
            # # **Confidence 기반 Cross-Attention Mask 생성**
            # confidence_mask = self.create_confidence_cross_attention_mask(
            #     confidence_scores, confidence_threshold
            # )
            # # Cross-attention mask를 corr_feats 차원에 맞게 확장
            # N, num_corrs, c, h, w = corr_feats.shape
            # # 신뢰도 낮은 영역을 직접 마스킹 (True=마스킹 대상)
            # cross_attn_mask = confidence_mask[:, None, None, None].expand(-1, num_corrs, h, w) # 객체단위 전체 masking

            # # 패딩 마스크와 신뢰도 마스크 결합 (and 연산)
            # padding_mask = ~mask[..., None, None].expand_as(corr_feats[:, :, 0])
            # combined_mask = padding_mask  | cross_attn_mask  # 패딩 or 낮은 신뢰도 → 마스킹

            ## **Modified bbox_head forward with confidence-based masking**
            # all_cls_scores, all_bbox_preds = self.bbox_head(ref_points[:, None],
            #                                                 corr_feats,  # confidence로 가중치가 적용된 features
            #                                                 padding_mask,        # 기존 mask + confidence mask
            #                                                 corr_pe,
            #                                                 attn_mask=None,
            #                                                 cross_attn_mask=None,  # confidence 기반 cross-attention mask
            #                                                 confidence_scores=confidence_scores,  # confidence_scores 추가
            #                                                 force_fp32=self.force_fp32,)

            all_cls_scores, all_bbox_preds = self.bbox_head(ref_points[:, None],
                                                            corr_feats,
                                                            ~mask[..., None, None].expand_as(corr_feats[:, :, 0]),
                                                            corr_pe,
                                                            attn_mask=None,
                                                            cross_attn_mask=None,
                                                            force_fp32=self.force_fp32, )
            
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]

        cls_scores, bbox_preds = [], []
        for c, b in zip(all_cls_scores, all_bbox_preds):
            cls_scores.append(c.flatten(0, 1))
            bbox_preds.append(b.flatten(0, 1))
        
        return_feats = {} ### 검증용
        bbox_results = dict(
            cls_scores=cls_scores, bbox_preds=bbox_preds, bbox_feats=bbox_feats, return_feats=return_feats,
            intrinsics=intrinsics, extrinsics=extrinsics, rois=rois, dn_mask_dict=mask_dict,
        )

        # bbox_results = dict(
        #     cls_scores=cls_scores, bbox_preds=bbox_preds, bbox_feats=bbox_feats, return_feats=return_feats,
        #     intrinsics=intrinsics, extrinsics=extrinsics, rois=rois, dn_mask_dict=mask_dict,
        #     conf_loss=conf_loss * self.conf_loss_weight,
        # )

        # return bbox_results , loss_corr
        return bbox_results

    # def _bbox_forward(self, x, proposal_list, img_metas): # for original 
    def _bbox_forward(self,img,img_metas,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): ### this modified moon
        # bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas) # for original 
        # bbox_results , loss_corr = self._bbox_forward_denoise(img,img_metas,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJMOON 
        bbox_results = self._bbox_forward_denoise(img,img_metas,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJMOON 
        # return bbox_results , loss_corr
        return bbox_results

    def prepare_for_dn_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt
    
    def _bbox_forward_train(self, img,img_metas,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): # for SJMOON
    # def _bbox_forward_train(self, x, proposal_list, img_metas): # for original 
        """Run forward function and calculate loss for box head in training."""
        # bbox_results , loss_corr = self._bbox_forward(img,img_metas,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJMOON
        bbox_results = self._bbox_forward(img,img_metas,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)
        # bbox_results = self._bbox_forward(x, proposal_list, img_metas) # for original 
        bbox_results.update(pred={'cls_scores': bbox_results['cls_scores'], 'bbox_preds': bbox_results['bbox_preds']})

        # return bbox_results, loss_corr
        return bbox_results

    def forward_train(self,
                      img,
                      img_metas,
                      lidar_depth_mis,
                      x,
                      proposal_list,
                      uvz_gt,
                      mis_KT,
                      mis_Rt,
                      gt_KT,
                      gt_KT_3by4,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      ori_gt_bboxes_3d,
                      ori_gt_labels_3d,
                      attr_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        num_imgs = len(img_metas)

        proposal_boxes = []
        proposal_scores = []
        proposal_classes = []
        for i in range(num_imgs):
            proposal_boxes.append(proposal_list[i][:, :6])
            proposal_scores.append(proposal_list[i][:, 4])
            proposal_classes.append(proposal_list[i][:, 5])

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        # mis_depth_pos_enc = self.position_encoding(mis_depthmap_feat, img_metas)
        # depth_x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(mis_depthmap_feat, mis_depth_pos_enc)]

        losses = dict()

        if self.use_denoise:
            img_metas[0]['gt_bboxes_3d'] = ori_gt_bboxes_3d[0]
            img_metas[0]['gt_labels_3d'] = ori_gt_labels_3d[0]

        # results_from_last , loss_corr = self._bbox_forward_train(img,img_metas,lidar_depth_mis, x, proposal_boxes, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJ MOON 
        results_from_last = self._bbox_forward_train(img,img_metas,lidar_depth_mis, x, proposal_boxes, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)
        # results_from_last = self._bbox_forward_train(x, proposal_boxes, img_metas) # for original 
        preds = results_from_last['pred']
        # Confidence 손실 추출
        # conf_loss = results_from_last['conf_loss']

        cls_scores = preds['cls_scores']
        bbox_preds = preds['bbox_preds']
        loss_weights = copy.deepcopy(self.stage_loss_weights)

        # Confidence 손실 추가 (가중치 0.5 적용 예시)
        # losses['conf_loss'] = conf_loss * 0.5  # 가중치는 실험적으로 조정
        # use the matching results from last stage for loss calculation
        loss_stage = []
        num_layers = len(cls_scores)
        for layer in range(num_layers):
            loss_bbox = self.bbox_head.loss(
                ori_gt_bboxes_3d, ori_gt_labels_3d, {'cls_scores': [cls_scores[num_layers - 1 - layer]],
                                                     'bbox_preds': [bbox_preds[num_layers - 1 - layer]]},
            )
            loss_stage.insert(0, loss_bbox)

        if results_from_last.get('dn_mask_dict', None) is not None:
            dn_mask_dict = results_from_last['dn_mask_dict']
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_dn_loss(
                dn_mask_dict)
            for i in range(len(output_known_class)):
                dn_loss_cls, dn_loss_bbox = self.bbox_head.dn_loss_single(
                    output_known_class[i], output_known_coord[i], known_bboxs, known_labels, num_tgt,
                    self.pc_range, self.denoise_split, neg_bbox_loss=self.neg_bbox_loss
                )
                losses[f'l{i}.dn_loss_cls'] = dn_loss_cls * self.denoise_weight * loss_weights[i]
                losses[f'l{i}.dn_loss_bbox'] = dn_loss_bbox * self.denoise_weight * loss_weights[i]

        for layer in range(num_layers):
            lw = loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v
        
        # losses['loss_corr'] = loss_corr
        # losses['total_loss_corr'] = total_loss_corr
        
        # return losses , loss_corr , loss_pc_distance
        return losses
    
    def simple_test(self,img,img_metas, lidar_depth_mis, x,proposal_list, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4,rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        # mis_depth_pos_enc = self.position_encoding(mis_depthmap_feat, img_metas)
        # depth_x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(mis_depthmap_feat, mis_depth_pos_enc)]

        results_from_last = dict()

        results_from_last['batch_size'] = len(img_metas) // img_metas[0]['num_views']
        # results_from_last ,_ = self._bbox_forward(img,img_metas, lidar_depth_mis,x,proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)
        results_from_last , _= self._bbox_forward(img,img_metas, lidar_depth_mis,x,proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)
        
        ## original
        cls_scores = results_from_last['cls_scores'][-1]
        bbox_preds = results_from_last['bbox_preds'][-1]

        # cls_scores = results_from_last[0]['cls_scores'][-1]
        # bbox_preds = results_from_last[0]['bbox_preds'][-1]

        bbox_list = self.bbox_head.get_bboxes({'cls_scores': [cls_scores], 'bbox_preds': [bbox_preds]}, img_metas,)

        return bbox_list

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
from mmdet.models.builder import HEADS , build_head
from .mv2d_head import MV2DHead
from COTR.COTR_models.cotr_model_moon_Ver12_0 import build
from torchvision.transforms import functional as tvtf
from torchvision.ops import DeformConv2d
from mmcv.ops import roi_align


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

class HeatmapHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: BEV feature map [B, C, H, W]
        heatmap = self.head(x)
        return heatmap  # [B, num_classes, H, W]
    
class FusionMLP(nn.Module):
    def __init__(self, in_features=6, out_features=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )
    def forward(self, img_ref, lidar_ref):
        # img_ref, lidar_ref: [N, 3]씩
        x = torch.cat([img_ref, lidar_ref], dim=1)  # [N, 6]
        return self.mlp(x)  # [N, 3]

@HEADS.register_module()
class MV2DSHead(MV2DHead):
    def __init__(self,
                 # denoise setting
                 voxelizer,
                 voxelnet,
                 corr,
                 corr_loss,
                 z_estimator,
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
        
        # self.num_kp =200
        # self.conf_loss_weight = 0.5
        # self.voxelization = SimpleVoxelization()
        # self.lidar_voxelnet = SimpleVoxelNet(load_pretrained_path='data/weights/second_7862.pth')
        # self.corr = COTR(self.num_kp) 
        # self.fine_corr = GraphBEVLocalAlignNet() 
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
        self.corr = build_head(corr)
        # self.corr_loss = build_head(corr_loss)
        self.z_estimator = build_head(z_estimator)

        self.voxelization = build_head(voxelizer)
        self.lidar_voxelnet = build_head(voxelnet)
        self.heat_map = HeatmapHead(in_channels=192, num_classes=10) #bev_feat.shape[1]
        self.fuse_mlp = FusionMLP(in_features=6, out_features=3)
       
        # self.z_estimator = ZEstimator(enc_channels=312, bbox_channels=256, uv_dim=2, hidden_dim=512)
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
    
    def get_reference_points_from_heatmap(self, heatmap, voxel_size, pc_range, top_k=100):
        """
        heatmap: [B, num_classes, H, W]
        voxel_size: (vx, vy)
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        B, num_classes, H, W = heatmap.shape
        batch_ref_points = []
        for b in range(B):
            heat_b = heatmap[b]  # [num_classes, H, W]
            scores, indices = torch.topk(heat_b.view(num_classes, -1), k=top_k, dim=1)
            # indices shape: [num_classes, top_k]

            ys = indices // W
            xs = indices % W

            # BEV 좌표 → 실제 좌표 변환
            x_min, y_min = pc_range[0], pc_range[1]
            vx, vy = voxel_size
            xs_real = xs * vx + x_min
            ys_real = ys * vy + y_min

            # reference points: (num_classes*top_k, 2), batch차원 포함할 수도 있음
            ref_points = torch.stack([xs_real, ys_real], dim=-1).reshape(-1, 2)
            batch_ref_points.append(ref_points)
        
        return batch_ref_points
    
    # def prepare_lidar_reference_points(self, lidar_reference_points, batch_size=1):
    #     """
    #     lidar_reference_points: list of tensor, 각 원소 shape = (num_proposals, 2)
    #     batch_size: 배치 크기

    #     반환 : tensor, shape = (batch_size, num_queries, 3)
    #     """

    #     # 1) list 내 각 batch 텐서 z 축 0 추가
    #     processed_list = []
    #     for points in lidar_reference_points:
    #         # points: (num_proposals, 2)
    #         num_points = points.shape[0]
    #         z = torch.zeros((num_points, 1), device=points.device, dtype=points.dtype)  # z=0 추가
    #         points_3d = torch.cat([points, z], dim=1)  # (num_proposals, 3)
    #         processed_list.append(points_3d)

    #     # 2) 리스트 텐서들 배치 차원으로 concat 또는 stack
    #     # 여기선 batch단위로 합치기 위해 pad or stack 시도 (간단히 배치=1 가정 시)
    #     # 만약 batch_size > 1이면 별도 로직 필요
    #     ref_points = torch.stack(processed_list, dim=0)  # (batch_size, num_proposals, 3)

    #     # 3) 배치 차원 맞추기 (원래 코드 대비) + query 차원 추가 (1)
    #     ref_points = ref_points.unsqueeze(2)  # (batch_size, num_proposals, 1, 3)

    #     return ref_points

    def fuse_reference_points(self,lidar_ref_point, ref_points):
        """
        lidar_ref_point: [num_lidar_candidates, 3]
        ref_points: [num_objects, 3]
        반환: fused_ref_points [num_objects, 3]
        """
        num_obj = ref_points.shape[0]
        num_lidar = lidar_ref_point.shape[0]

        # 모든 객체에 대해 lidar 후보와의 거리를 구함
        # 거리: [num_objects, num_lidar_candidates]
        distances = torch.norm(ref_points[:, None, :] - lidar_ref_point[None, :, :], dim=2) 

        # 각 object 별로 가장 가까운 lidar candidate 추출
        closest_indices = distances.argmin(dim=1)  # [num_objects]
        matched_lidar_points = lidar_ref_point[closest_indices]  # [num_objects, 3]

        # 임베딩 fusion: 간단히 평균, 혹은 weighted sum/MLP로 가능
        # fused_ref_points = 0.5 * ref_points + 0.5 * matched_lidar_points

        # 더 고급: torch.cat 후 linear/MLP/attention으로 임베딩 융합 가능
        # fused_ref_points = fusion_mlp(torch.cat([ref_points, matched_lidar_points], dim=1))  # [num_objects, 3]

        return matched_lidar_points
    
    def fuse_reference_points_guided_k(self, image_ref, lidar_ref, k=3, alpha=0.8):
        num_objects = image_ref.shape[0]
        fused = torch.zeros_like(image_ref)
        
        support_points_all = [] 
        for i in range(num_objects):
            guide_point = image_ref[i]  # (3,)
            # 1) 거리 정렬 후 k개 support 선택
            distances = torch.norm(lidar_ref - guide_point, dim=1)
            if lidar_ref.shape[0] < k:
                support_points = lidar_ref
            else:
                topk_indices = distances.topk(k, largest=False).indices  # 가까운 k
                support_points = lidar_ref[topk_indices]  # [k,3]
            # 2) k-support의 평균
            support_point = support_points.mean(dim=0)
            support_points_all.append(support_point)

            # 3) 가중 평균
            fused[i] = alpha * guide_point + (1-alpha) * support_point
        # [num_objects, 3]로 변환
        support_points_tensor = torch.stack(support_points_all, dim=0)    
        return support_points_tensor, fused
    
    def create_proposals_from_ref_points(self, ref_points, box_size=4):
        """
        ref_points: tensor, shape [num_points, 3] (x, y, z)
        box_size: float, BEV bbox 한 변의 길이 (m 단위)

        Returns:
            proposals: list of tensors, [num_points, 4], 각 2D bbox 좌표 [x1,y1,x2,y2] 
        """
        half_size = box_size / 2
        x = ref_points[:, 0]
        y = ref_points[:, 1]

        x1 = x - half_size
        y1 = y - half_size
        x2 = x + half_size
        y2 = y + half_size

        proposals = torch.stack([x1, y1, x2, y2], dim=1)  # [num_points, 4]

        # 리스트 포맷으로 반환, batch 처리용 (batch size=1 가정)
        return [proposals]

    def roi_align_custom(self, bev_feat, proposals, output_size=7, pc_range=None):
        # pc_range가 반드시 전달되어야 함: [x_min, y_min, z_min, x_max, y_max, z_max]
        # bev_feat.shape = [B, C, H, W]
        B, C, H, W = bev_feat.shape

        rois = []
        for batch_idx, b_proposals in enumerate(proposals):
            if b_proposals.numel() == 0:
                continue

            # 실세계 좌표(b_proposals: [num_rois, 4]) → 픽셀 좌표로 변환
            # x1, y1, x2, y2 각각 변환
            b_proposals_pixel = b_proposals.clone()
            x_min, y_min, _, x_max, y_max, _ = pc_range

            b_proposals_pixel[:, 0] = (b_proposals[:, 0] - x_min) / (x_max - x_min) * W  # x1
            b_proposals_pixel[:, 1] = (b_proposals[:, 1] - y_min) / (y_max - y_min) * H  # y1
            b_proposals_pixel[:, 2] = (b_proposals[:, 2] - x_min) / (x_max - x_min) * W  # x2
            b_proposals_pixel[:, 3] = (b_proposals[:, 3] - y_min) / (y_max - y_min) * H  # y2

            batch_idx_tensor = b_proposals_pixel.new_full((b_proposals_pixel.size(0), 1), batch_idx)
            rois_batch = torch.cat([batch_idx_tensor, b_proposals_pixel], dim=1)  # [N,5]
            rois.append(rois_batch)

        if len(rois) == 0:
            return None  # ROI가 없는 경우 처리

        rois = torch.cat(rois, dim=0)  # 리스트를 하나의 텐서로 변환

        roi_feats = roi_align(bev_feat, rois, (output_size, output_size))

        return roi_feats


    def _bbox_forward_denoise(self, img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): # for SJMOON
    # def _bbox_forward_denoise(self, x, proposal_list, img_metas): # for original 
        # avoid empty 2D detection
        with torch.no_grad():
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
            # dense_depth_map_gt = dense_map_from_depth_batch(uvz_gt.squeeze(0),grid=3,iterations=3)
            dense_depth_map = dense_map_from_depth_batch(lidar_depth_mis,grid=3,iterations=3)
            dense_depth_img_mis = dense_depth_map.to(dtype=torch.uint8)
            dense_depth_img_color_mis = batch_colormap(dense_depth_img_mis)
            # dense_depth_img_color_mis = differentiable_colormap(dense_depth_img_mis)
            
            img_resized = F.interpolate(img, size=[192, 640], mode="bilinear")
            lidar_depth_mis_resized = F.interpolate(dense_depth_img_color_mis, size=[192, 640], mode="bilinear")
            lidar_depth_mis_resized = tvtf.normalize(lidar_depth_mis_resized, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            lidar_depth_mis_resized = lidar_depth_mis_resized.to(img_resized.dtype)
            # lidar_depth_mis_resized = F.interpolate(lidar_depth_mis, size=[h, w], mode="bilinear")
            
            # Deformable SPN 적용 lidar_depth_mis_resized = lidar_depth_mis_resized.to(img_resized.dtype)(주요 수정 부분)
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
            # trimed_uvset = batched_trim_corrs(uv_set).to(dtype=torch.float32, device=img.device)
            # 객체 ID 보존 텐서
            object_ids = trimed_center_pts[..., 1].clone()  # [num_cams, batch_size]

            # 쿼리 입력 생성 (좌표만 정규화)
            # query_input = trimed_uvset[..., :2]
            query_input = trimed_center_pts[..., 2:].clone()  # [num_cams, batch_size, 2]
            # scaled1_query_input = scale_uvz_points(query_input,original_size=(900,1600),target_size=(192,640))
            # scaled2_query_input = normalize_uv_points(scaled1_query_input)

            query_input[..., 0] /= img.shape[3]  # 1600
            query_input[..., 1] /= img.shape[2]  # 928
            query_input[:,:,0] = query_input[:,:,0]/2    # recaling points for sbs image resizing
            query_input[:,:,1] = query_input[:,:,1]

            # corr_target = trimed_uvset[...,2:]
            # corr_target[...,0] = corr_target[...,0] / 1408.0
            # corr_target[...,1] = corr_target[...,1] / 512.0
            # corr_target[:,:,0] = corr_target[:,:,0]/2 + 0.5 # recaling points for sbs image resizing
            # corr_target[:,:,1] = corr_target[:,:,1] 

            raw_corrs, cycle, corr_mask, enc_out = self.corr(sbs_img, query_input)
            # 객체 ID 정보를 예측 결과에 연결

            # loss_corr = self.corr_loss(raw_corrs, corr_target, cycle, query_input, corr_mask)
            # fine_raw_corrs = self.fine_corr(raw_corrs, dense_depth_map)
            # fine_raw_corrs[...,0] = fine_raw_corrs[...,0] - 0.5
            # loss_corr = self.corr_loss(fine_raw_corrs[...,:2], query_input, img, dense_depth_map)
            
            corrs_pred_with_obj = torch.cat([
                object_ids.unsqueeze(-1),  # [num_cams, batch_size, 1]
                raw_corrs                 # [num_cams, batch_size, 2]
            ], dim=-1)  # [num_cams, batch_size, 3]
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
            raw_pred_center_pts2[..., 2] *= dense_depth_img_color_mis.shape[3]  # 1600
            raw_pred_center_pts2[..., 3] *= dense_depth_img_color_mis.shape[2]  # 900

            # # ##### 검증용 display ######
            # from image_processing_unit_Ver15_0 import draw_correspondences
            # # corrs_pred_norm = self.inverse_layer_norm(corrs_pred, self.final_ln)
            # # rois_center_disp = scale_uvz_points(rois_center[...,2:],original_size=(900,1600),target_size=(192,640))
            # # trimed_corrs = batch_rois_center_by_cam_id(rois_center,batch_size=200)
            # # pred_corrs = torch.cat([rois_center_disp,pred_center_pts1[...,2:]],dim=-1)
            # # gt_corrs = torch.cat([query_input,corr_target],dim=-1)
            # pred_corrs = torch.cat([query_input,raw_corrs],dim=-1)
            # # int_ids = original_camera_ids.to(torch.long).cpu()
            # # if len(int_ids) < 6:
            # #     print ("len(int_ids) < 6")
            # # 카메라 ID ↔ 인덱스 매핑 생성
            # # id_to_idx = {cid.item(): idx for idx, cid in enumerate(original_camera_ids)}
            # # for cid in int_ids :
            # for cid in range(6):
            #     # idx = id_to_idx[cid.item()]
            #     # draw_correspondences(
            #     #     trimed_corrs = gt_corrs[cid][:10,...],  # 첫 번째 배치 선택
            #     #     sbs_img=sbs_img[cid],
            #     #     save_path='correspondence_visualization_gt.jpg'
            #     # )
            #     draw_correspondences(
            #         trimed_corrs = pred_corrs[cid][:1,...],  # 첫 번째 배치 선택
            #         sbs_img=sbs_img[cid],
            #         save_path='correspondence_visualization_pred.jpg'
            #     )
            #     print ("end")

            # transformed_uv = transform_uv_points(rois_with_indices,uv_set)      
            # esitmated_z = self.z_estimator(transformed_uv[...,:4], dense_depth_map_gt,bbox_feats,ref_points_uvz)
            # esitmated_z = self.z_estimator(pred_center_pts, dense_depth_map_gt,bbox_feats)
            esitmated_z = self.z_estimator(raw_pred_center_pts2, dense_depth_map,bbox_feats, enc_out)
            esitmated_uvz =torch.cat([raw_pred_center_pts2[...,:4], esitmated_z['depth']],dim=1)

            detection_nonzero_xyz = image_to_lidar_global_modi3(esitmated_uvz,gt_KT) # 교정되어진 lidar좌표계 pc

            detection_xyz = detection_nonzero_xyz[:,2:5].clone()
            detection_xyz_normalized = detection_xyz.clone()

            detection_xyz_normalized[..., 0:1] = (detection_xyz_normalized[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
            detection_xyz_normalized[..., 1:2] = (detection_xyz_normalized[..., 1:2] - self.pc_range[1]) / (
                    self.pc_range[4] - self.pc_range[1])
            detection_xyz_normalized[..., 2:3] = (detection_xyz_normalized[..., 2:3] - self.pc_range[2]) / (
                    self.pc_range[5] - self.pc_range[2])
            detection_xyz_normalized = detection_xyz_normalized.clamp(min=0, max=1)
            
            ref_points = detection_xyz_normalized
            
        #### voxelization ######
        # with torch.no_grad():
        pts_voxels,pts_coords,pts_num_points = self.voxelization(raw_points)
        bev_feat = self.lidar_voxelnet(pts_voxels, pts_coords, pts_num_points)
        _, _, H, W = bev_feat[1].shape  # [batch, channel, h, w]
        bev_feat0_resize = F.interpolate(bev_feat[0], size=(H, W), mode='bilinear', align_corners=False)
        mixed_bev_feat = torch.cat([bev_feat0_resize, bev_feat[1]], dim=1)
        # bev_feat_1 = bev_feat[1]  # [B, C, H, W]
        heatmap = self.heat_map(mixed_bev_feat)
        xy_ref_point =self.get_reference_points_from_heatmap(heatmap,voxel_size=(0.2, 0.2), pc_range=[0, -40, -3, 70.4, 40, 1], top_k=10)
        dummy_z = torch.zeros((100, 1), device=mixed_bev_feat.device, dtype=mixed_bev_feat.dtype)  # z=0 추가
        lidar_reference_point = torch.cat([xy_ref_point[0], dummy_z], dim=1)  # (num_proposals, 3)

        # 3D proposal 후보 생성
        proposals = self.create_proposals_from_ref_points(lidar_reference_point, box_size=4)

        # proposal 기반 RoI Align 피쳐 추출
        bev_roi_feats = self.roi_align_custom(mixed_bev_feat, proposals, output_size=7, pc_range=[0, -40, -3, 70.4, 40, 1])

        lidar_ref_point = lidar_reference_point.clone()
        lidar_ref_point[..., 0:1] = (lidar_ref_point[..., 0:1] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0])
        lidar_ref_point[..., 1:2] = (lidar_ref_point[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        lidar_ref_point[..., 2:3] = (lidar_ref_point[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        lidar_ref_point = lidar_ref_point.clamp(min=0, max=1)
        
        # # matched_lidar_points = self.fuse_reference_points(lidar_ref_point, ref_points)
        # matched_lidar_points , fused = self.fuse_reference_points_guided_k(ref_points,lidar_ref_point)
        # # fused_ref_points = self.fuse_mlp(ref_points, matched_lidar_points)
        # fused_ref_points = fused

        # generate box correlation
        corr, mask = self.box_corr_module.gen_box_roi_correlation(rois, [len(p) for p in proposal_list], img_metas)

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

            # corr_feats = bbox_feats[corr]  # [num_rois, num_corrs, c, h, w]
            # corr_pe = pe[corr]

            # --- 수정된 코드 ---
            corr_feats_list = []
            corr_pe_list = []
            num_rois = corr.shape[0]

            # RoI를 하나씩 또는 작은 그룹으로 나누어 처리
            for i in range(num_rois):
                # i번째 RoI에 해당하는 corr 인덱스만 사용
                current_corr_indices = corr[i]
                # 유효한 인덱스만 필터링 (패딩 인덱스 제외)
                valid_mask = current_corr_indices < bbox_feats.shape[0]
                valid_indices = current_corr_indices[valid_mask]
                
                # 작은 단위로 피처 추출
                temp_corr_feats = bbox_feats[valid_indices]
                temp_corr_pe = pe[valid_indices]
                
                # 패딩 처리 (필요한 경우)
                if temp_corr_feats.shape[0] < corr.shape[1]:
                    pad_size = corr.shape[1] - temp_corr_feats.shape[0]
                    padding_feats = torch.zeros(pad_size, *temp_corr_feats.shape[1:], device=bbox_feats.device, dtype=bbox_feats.dtype)
                    padding_pe = torch.zeros(pad_size, *temp_corr_pe.shape[1:], device=pe.device, dtype=pe.dtype)
                    temp_corr_feats = torch.cat([temp_corr_feats, padding_feats], dim=0)
                    temp_corr_pe = torch.cat([temp_corr_pe, padding_pe], dim=0)

                corr_feats_list.append(temp_corr_feats)
                corr_pe_list.append(temp_corr_pe)

            # 작은 결과들을 마지막에 하나로 합침
            corr_feats = torch.stack(corr_feats_list, dim=0)
            corr_pe = torch.stack(corr_pe_list, dim=0)
            # bev_input = bev_feat_1[:, None]
            bev_input = bev_roi_feats[:, None]
            all_cls_scores, all_bbox_preds = self.bbox_head(ref_points[:, None],
                                                            corr_feats,
                                                            ~mask[..., None, None].expand_as(corr_feats[:, :, 0]),
                                                            corr_pe,
                                                            bev_input,
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
    def _bbox_forward(self,img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): ### this modified moon
        # bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas) # for original 
        # bbox_results , loss_corr = self._bbox_forward_denoise(img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJMOON 
        bbox_results = self._bbox_forward_denoise(img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJMOON 
        # return bbox_results , loss_corr
        torch.cuda.empty_cache()
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
    
    def _bbox_forward_train(self, img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): # for SJMOON
    # def _bbox_forward_train(self, x, proposal_list, img_metas): # for original 
        """Run forward function and calculate loss for box head in training."""
        # bbox_results , loss_corr = self._bbox_forward(img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJMOON
        bbox_results = self._bbox_forward(img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)
        # bbox_results = self._bbox_forward(x, proposal_list, img_metas) # for original 
        bbox_results.update(pred={'cls_scores': bbox_results['cls_scores'], 'bbox_preds': bbox_results['bbox_preds']})

        # return bbox_results, loss_corr
        return bbox_results

    def forward_train(self,
                      img,
                      img_metas,
                      raw_points,
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

        # results_from_last , loss_corr = self._bbox_forward_train(img,img_metas,raw_points,lidar_depth_mis, x, proposal_boxes, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJ MOON 
        results_from_last = self._bbox_forward_train(img,img_metas,raw_points,lidar_depth_mis, x, proposal_boxes, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)
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

        ### 3d bbox loss insert ######
        for layer in range(num_layers):
            lw = loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v
        
        # losses['loss_corr'] = loss_corr
        # losses['total_loss_corr'] = total_loss_corr
        
        # return losses , loss_corr , loss_pc_distance
        return losses
    
    def simple_test(self,img,img_metas,raw_points, lidar_depth_mis, x,proposal_list, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4,rescale=False):
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
        results_from_last = self._bbox_forward(img,img_metas,raw_points, lidar_depth_mis,x,proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)
        
        ## original
        cls_scores = results_from_last['cls_scores'][-1]
        bbox_preds = results_from_last['bbox_preds'][-1]

        # cls_scores = results_from_last[0]['cls_scores'][-1]
        # bbox_preds = results_from_last[0]['bbox_preds'][-1]

        bbox_list = self.bbox_head.get_bboxes({'cls_scores': [cls_scores], 'bbox_preds': [bbox_preds]}, img_metas,)

        return bbox_list

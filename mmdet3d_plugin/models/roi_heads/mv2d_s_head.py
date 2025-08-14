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


@HEADS.register_module()
class MV2DSHead(MV2DHead):
    def __init__(self,
                 # denoise setting
                #  voxelizer,
                #  voxelnet,
                #  corr,
                #  corr_loss,
                #  z_estimator,
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
        # self.corr = build_head(corr)
        # self.corr_loss = build_head(corr_loss)
        # self.z_estimator = build_head(z_estimator)

        # self.voxelization = build_head(voxelizer)
        # self.lidar_voxelnet = build_head(voxelnet)
       
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

    def _bbox_forward_denoise(self, img,img_metas,raw_points,lidar_depth_mis,x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): # for SJMOON
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

        # #### voxelization ######
        # pts_voxels,pts_coords,pts_num_points = self.voxelization(raw_points)
        # bev_feat = self.lidar_voxelnet(pts_voxels, pts_coords, pts_num_points)

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
        # trimed_uvset = batched_trim_corrs(uv_set).to(dtype=torch.float32, device=img.device)
        # 객체 ID 보존 텐서
        object_ids = trimed_center_pts[..., 1].clone()  # [num_cams, batch_size]

        # 쿼리 입력 생성 (좌표만 정규화)
        # query_input = trimed_uvset[..., :2]
        query_input = trimed_center_pts[..., 2:].clone()  # [num_cams, batch_size, 2]
        # scaled1_query_input = scale_uvz_points(query_input,original_size=(900,1600),target_size=(192,640))
        # scaled2_query_input = normalize_uv_points(scaled1_query_input)

        query_input[..., 0] /= 1408.0
        query_input[..., 1] /= 512.0
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
        raw_pred_center_pts2[..., 2] *= 1408.0
        raw_pred_center_pts2[..., 3] *= 512.0

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
        #         trimed_corrs = pred_corrs[cid][:10,...],  # 첫 번째 배치 선택
        #         sbs_img=sbs_img[cid],
        #         save_path='correspondence_visualization_pred.jpg'
        #     )
        #     print ("end")

        # transformed_uv = transform_uv_points(rois_with_indices,uv_set)      
        # esitmated_z = self.z_estimator(transformed_uv[...,:4], dense_depth_map_gt,bbox_feats,ref_points_uvz)
        # esitmated_z = self.z_estimator(pred_center_pts, dense_depth_map_gt,bbox_feats)
        esitmated_z = self.z_estimator(raw_pred_center_pts2, dense_depth_map,bbox_feats, enc_out)
        # **Confidence 정보 추출**
        # confidence_scores = esitmated_z['confidence'].view(-1,1)  # [N]
        # z_depth_real = esitmated_z['z_lidar_real']  # [N]
        # fine_z_raw = raw_pred_center_pts2[..., 4].reshape(-1, 1)  # [N, 1]
        # 융합된 z 계산 (예: confidence 가중 평균)
        # z_fused = confidence_scores * fine_z_raw + (1 - confidence_scores) * esitmated_z['depth']  # [N, 1]
        esitmated_uvz =torch.cat([raw_pred_center_pts2[...,:4], esitmated_z['depth']],dim=1)

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
                                                            bev_feat=None,
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

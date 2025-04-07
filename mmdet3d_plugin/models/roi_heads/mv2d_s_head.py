# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys

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
                                           find_rois_nonzero_z_adv6,find_rois_nonzero_z_adv7,
                                           image_to_lidar_global,image_to_lidar_global_modi,image_to_lidar_global_modi1,
                                           lidar_to_image_with_index,
                                           miscalib_transform, miscalib_transform1,miscalib_transform2,
                                           points2depthmap,dense_map_gpu_optimized,colormap,
                                           two_images_side_by_side,
                                           display_depth_maps,scale_uvz_points,normalize_uvz_points,
                                           inverse_scale_uvz_points,
                                           trim_corrs,denormalize_points,process_queries,process_queries_adv,
                                           selected_image_to_lidar_global,
                                           pixel_to_normalized,center2lidar_batch,
                                           project_lidar_to_image,minmax_normalize_uvz,minmax_denormalize_uvz,
                                           draw_corrs , lidar_to_image_no_filter,
                                           geometric_propagation)

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
            mask = torch.norm(cycle - query_input, dim=-1) < 10 / 640 # 40 pixel 거리에서는 마스크 

        return corrs_pred , cycle , mask , enc_out
    

class CorrelationCycleLoss(nn.Module):
    def __init__(self, corr_weight=1.0 , cycle_weight=1.0):
        super().__init__()
        # self.loss_weight = loss_weight
        self.corr_weight = corr_weight
        self.cycle_weight= cycle_weight
        # self.chamfer_weight = corr_weight
        # print ("corr weights=" , self.loss_weight )

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
    
    # def forward(self, corr_pred, corr_target, cycle, queries, mask):
    #     # Learnable Chamfer Distance 구현
    #     def chamfer_distance(x, y):
    #         # x, y: 배치 단위 포인트 클라우드 (B, N, D)
    #         x_size = x.size(1)
    #         y_size = y.size(1)
            
    #         # 효율적인 거리 계산을 위한 방식
    #         x = x.unsqueeze(2).expand(-1, -1, y_size, -1)  # (B, N, M, D)
    #         y = y.unsqueeze(1).expand(-1, x_size, -1, -1)  # (B, N, M, D)
            
    #         # L2 거리 계산
    #         dist = torch.pow(x - y, 2).sum(3)  # (B, N, M)
            
    #         # 양방향 최소 거리 계산
    #         min_dist_xy = dist.min(2)[0]  # (B, N)
    #         min_dist_yx = dist.min(1)[0]  # (B, M)
            
    #         # 가중치 네트워크를 통한 적응형 가중치 부여
    #         # 간단한 구현을 위해 일반 가중치 사용
    #         chamfer_loss = min_dist_xy.mean() + min_dist_yx.mean()
            
    #         return chamfer_loss
    
    #     # 기본 Chamfer 손실
    #     chamfer_loss = chamfer_distance(corr_pred, corr_target)
        
    #     # Cycle Consistency 손실
    #     cycle_loss = torch.tensor(0.0, device=chamfer_loss.device)
    #     if mask.sum() > 0:
    #         cycle_loss = torch.nn.functional.smooth_l1_loss(cycle[mask], queries[mask])
        
    #     # 추가적인 Point Cloud Distance 손실 (선택적)
    #     # point_dist_loss = point_distance_loss(corr_pred, corr_target)
    #     corr_loss = self.chamfer_weight * chamfer_loss + self.cycle_weight * cycle_loss
    
    #     return corr_loss

# Deformable SPN 모듈 정의
class DeformableSPN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 18, 3, padding=1)  # 3x3 커널 기준 2*9 offset
        )
        self.dcn = DeformConv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        offsets = self.offset_conv(x)
        return self.dcn(x, offsets)

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
        
        # # ##### refernce points 가변 레이어 설정 #########
        # # dynamic_linear = nn.Linear(batch_size*seq_len*num_points*feature_dim , num_points * 3)
        # # # 가중치 초기화
        # self.weight = nn.Parameter(torch.Tensor(6 * 1 *500*256 , 500 * 3))
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # # # 편향 초기화
        # self.bias = nn.Parameter(torch.Tensor(500 * 3))
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # bound = 1 / math.sqrt(fan_in)
        # nn.init.uniform_(self.bias, -bound, bound)

        # self.max_points = 500
        # self.feature_dim = 256
        
        # # 초기 가중치와 편향 생성 (작은 크기로 시작)
        # self.weight = nn.Parameter(torch.Tensor(self.feature_dim, 3))
        # self.bias = nn.Parameter(torch.Tensor(3))
        # # 가중치 초기화
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # # 편향 초기화
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # bound = 1 / math.sqrt(fan_in)
        # nn.init.uniform_(self.bias, -bound, bound)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        # self.flatten = nn.Flatten()
        # self.mlp = nn.Sequential(
        #     nn.Linear(912, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3)
        # )
        self.corr_loss = CorrelationCycleLoss(corr_weight=2.0 , cycle_weight=1.0)
        
        self.num_kp = 100 
        self.corr = COTR(self.num_kp)


        # 레이어 정규화 적용 (각 포인트 독립 정규화)
        # self.final_ln = nn.LayerNorm(3)  # C: 특징 차원 (x,y,z 등)
        # self.final_ln_sparse_cross_attn = nn.LayerNorm(3)  # C: 특징 차원 (x,y,z 등)

        # # 분포 추적 버퍼 초기화 (모델 클래스 내부에 선언)
        # if not hasattr(self, 'corr_stats'):
        #     self.register_buffer('corr_mean', torch.zeros(3))
        #     self.register_buffer('corr_std', torch.ones(3))

        # self.empty_count = 0  # 빈 텐서 카운터 초기화
        # self.total_iter = 0   # 전체 이터레이션 카운터

        # Deformable SPN 레이어 추가
        self.deform_spn = DeformableSPN()

      
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

    def _bbox_forward_denoise(self, img,img_metas,lidar_depth_mis,x, depth_x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): # for SJMOON
    # def _bbox_forward_denoise(self, x, proposal_list, img_metas): # for original 
        # avoid empty 2D detection
        if sum([len(p) for p in proposal_list]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposal_list[0].dtype,
                                    device=proposal_list[0].device)
            proposal_list = [proposal] + proposal_list[1:]

        rois = bbox2roi(proposal_list)
        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # 3dpe was concatenated to fpn feature
        c = bbox_feats.size(1)
        bbox_feats, pe = bbox_feats.split([c // 2, c // 2], dim=1)

        # intrinsics as extra input feature
        extra_feats = dict(
            intrinsic=self.process_intrins_feat(rois, intrinsics)
        )

        ###### SJ MOON 수정 #############
        img_resized = F.interpolate(img, size=[192, 640], mode="bilinear")
        lidar_depth_mis_resized = F.interpolate(lidar_depth_mis, size=[192, 640], mode="bilinear")
        # lidar_depth_mis_resized = F.interpolate(lidar_depth_mis, size=[h, w], mode="bilinear")

        # Deformable SPN 적용 (주요 수정 부분)
        # dense_depth = self.deform_spn(lidar_depth_mis_resized)
        # dense_depth = geometric_propagation(lidar_depth_mis_resized)

        sbs_img = two_images_side_by_side(img_resized, lidar_depth_mis_resized)
        sbs_img = torch.from_numpy(sbs_img).permute(0,3,1,2).to('cuda')
        sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ############## input display ##########################
        # display_depth_maps(img,lidar_depth_mis,sbs_img)
        
        # ### query generator
        # ref_points_uvz,reference_points,return_feats = self.query_generator(bbox_feats, intrinsics, extrinsics, extra_feats)
        # reference_points_raw = reference_points.clone().detach()
        # reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (
        #         self.pc_range[3] - self.pc_range[0])
        # reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (
        #         self.pc_range[4] - self.pc_range[1])
        # reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (
        #         self.pc_range[5] - self.pc_range[2])
        # reference_points = reference_points.clamp(min=0, max=1)

        # detection_nonzero_uvz , conf_scores = find_rois_nonzero_z_adv4(rois,uvz_gt,ref_points_uvz)
        # detection_nonzero_uvz = find_rois_nonzero_z_adv5(rois,uvz_gt,ref_points_uvz)
        # detection_nonzero_uvz_with_ObjectID = find_rois_nonzero_z_adv6(rois,uvz_gt,ref_points_uvz)
        detection_nonzero_uvz_with_ObjectID =find_rois_nonzero_z_adv7(rois,uvz_gt)
        # detection_nonzero_uvz = detection_nonzero_uvz_with_ObjectID[:,1:]
        # pixel_normal_uvz = pixel_to_normalized(detection_nonzero_uvz,intrinsics)
        # detection_xyz_adv ,lidar2img = center2lidar(pixel_normal_uvz[:,1:4],intrinsics,extrinsics)
        # detection_xyz_adv_concat = torch.cat([pixel_normal_uvz[:,0:1],detection_xyz_adv,pixel_normal_uvz[:,4:5]],dim=1)
        # detection_nonzero_xyz = detection_xyz_adv_concat.float()
        detection_nonzero_xyz = image_to_lidar_global_modi1(detection_nonzero_uvz_with_ObjectID,gt_KT) # 교정되어진 lidar좌표계 pc
        detection_real_mask = (detection_nonzero_uvz_with_ObjectID[:,5] == 1.0) # | (detection_nonzero_xyz[:,4] == 0.8)
        # detection_pred_mask = (detection_nonzero_uvz_with_ObjectID[:,5] == 0.7)
        detection_uvz_lidar = detection_nonzero_uvz_with_ObjectID[detection_real_mask]
        detection_xyz_lidar = detection_nonzero_xyz[detection_real_mask]
        # detection_xyz_pred = detection_nonzero_xyz[detection_pred_mask]
        
        gt_xyz = miscalib_transform2(detection_nonzero_xyz,mis_Rt)
        gt_xyz_lidar = gt_xyz[detection_real_mask]

        detection_xyz = detection_xyz_lidar[:,2:5].clone()
        pts_lidar_mis = gt_xyz_lidar[:,2:5].clone()

        # detection_xyz_total = detection_nonzero_xyz[:,1:4].clone()
        # gt_xyz_total = gt_xyz[:,1:4].clone()

        ## homogeous - with index
        pts_hom = torch.cat([pts_lidar_mis, torch.ones_like(pts_lidar_mis[:, :1])], dim=1)
        pts_hom_with_index = torch.cat([gt_xyz_lidar[:,0:1],pts_hom],dim=1)
        det_xyz_hom = torch.cat([detection_xyz, torch.ones_like(detection_xyz[:, :1])], dim=1)
        det_xyz_hom_with_index = torch.cat([detection_xyz_lidar[:,0:1],det_xyz_hom],dim=1)
        
        # points_lidar2img_mis = project_lidar_to_image(pts_hom,lidar2img)
        # points_lidar2img = project_lidar_to_image(det_xyz_hom,lidar2img)
        points_lidar2img_mis ,mask_valid_mis = lidar_to_image_with_index(pts_hom_with_index,gt_KT)
        points_lidar2img = lidar_to_image_no_filter(det_xyz_hom_with_index,gt_KT)
        points_lidar2img = points_lidar2img[mask_valid_mis]

        scaled_points_lidar2img_mis = scale_uvz_points(points_lidar2img_mis[:,1:])
        scaled_points_lidar2img = scale_uvz_points(points_lidar2img[:,1:])

        # normal_points_lidar2img_mis , mis_min_vals, mis_max_vals= minmax_normalize_uvz(points_lidar2img_mis)
        # normal_points_lidar2img_mis[:, 0] += 0.5
        # normal_points_lidar2img , min_vals, max_vals  = minmax_normalize_uvz(points_lidar2img)

        normal_points_lidar2img_mis = normalize_uvz_points(scaled_points_lidar2img_mis)
        # modified_points = normal_points_lidar2img_mis.clone()
        normal_points_lidar2img_mis[:, 0] += 0.5
        normal_points_lidar2img = normalize_uvz_points(scaled_points_lidar2img)
        corrs_points = torch.cat([detection_xyz_lidar[mask_valid_mis][:,0:2],normal_points_lidar2img,normal_points_lidar2img_mis],dim=1)
        
        ######## 검증용 corrs display ########
        # list_trimed_corrs =[]
        # for camera_idx in range(6):
        #     mask = pts_hom_with_index[:, 0] == camera_idx
        #     lidar_points = pts_hom_with_index[mask, 1:5]
        #     veri_points_lidar2img_mis = (gt_KT[camera_idx] @ lidar_points.T).T
        #     veri_points_lidar2img_mis = torch.cat([veri_points_lidar2img_mis[:, :2] / veri_points_lidar2img_mis[:, 2:3], veri_points_lidar2img_mis[:, 2:3]], 1)

        #     lidar_points = det_xyz_hom_with_index[mask, 1:5]
        #     veri_points_lidar2img = (gt_KT[camera_idx] @ lidar_points.T).T
        #     veri_points_lidar2img = torch.cat([veri_points_lidar2img[:, :2] / veri_points_lidar2img[:, 2:3], veri_points_lidar2img[:, 2:3]], 1)
            
        #     veri_scaled_points_lidar2img_mis = scale_uvz_points(veri_points_lidar2img_mis)
        #     veri_scaled_points_lidar2img = scale_uvz_points(veri_points_lidar2img)
        #     veri_normal_points_lidar2img_mis = veri_scaled_points_lidar2img_mis.clone()
        #     veri_normal_points_lidar2img = veri_scaled_points_lidar2img.clone()

        #     veri_normal_points_lidar2img_mis[:, 0] = veri_normal_points_lidar2img_mis[:, 0]/640
        #     veri_normal_points_lidar2img_mis[:, 1] = veri_normal_points_lidar2img_mis[:, 1]/192 
        #     if veri_normal_points_lidar2img_mis[:, 2].numel() > 0:
        #         veri_normal_points_lidar2img_mis[:, 2] = (veri_normal_points_lidar2img_mis[:, 2]-torch.min(veri_normal_points_lidar2img_mis[:, 2]))\
        #             /(torch.max(veri_normal_points_lidar2img_mis[:, 2]) - torch.min(veri_normal_points_lidar2img_mis[:, 2]))
        #     else :
        #         veri_normal_points_lidar2img_mis[:, 2] = (veri_normal_points_lidar2img_mis[:, 2]-0)/(80 - 0)
        #     veri_normal_points_lidar2img_mis[:, 0] += 0.5
            
        #     veri_normal_points_lidar2img[:, 0] = veri_normal_points_lidar2img[:, 0]/640
        #     veri_normal_points_lidar2img[:, 1] = veri_normal_points_lidar2img[:, 1]/192
        #     if veri_normal_points_lidar2img[:, 2].numel() > 0:
        #         veri_normal_points_lidar2img[:, 2] = (veri_normal_points_lidar2img[:, 2]-torch.min(veri_normal_points_lidar2img[:, 2]))\
        #             /(torch.max(veri_normal_points_lidar2img[:, 2]) - torch.min(veri_normal_points_lidar2img[:, 2]))
        #     else :
        #         veri_normal_points_lidar2img[:, 2] = (veri_normal_points_lidar2img[:, 2]-0)/(80 - 0)
            
        #     points_corrs= torch.cat([veri_normal_points_lidar2img,veri_normal_points_lidar2img_mis],dim=1)
        #     trimed_corrs = trim_corrs(points_corrs)
            
        #     list_trimed_corrs.append(trimed_corrs)
        #     #### corrspondence points display ######
        #     import matplotlib.pyplot as plt
        #     # 입력 이미지 처리
        #     img_tensor = sbs_img[camera_idx]  # [3, 192, 1280]
        #     denorm_img = img_tensor / 2 + 0.5  # 정규화 해제
        #     img_np = denorm_img.permute(1, 2, 0).cpu().numpy()

        #     # 3차원 좌표에서 2D 이미지 좌표 추출 (z값 제거)
        #     left_pts = veri_scaled_points_lidar2img.cpu().numpy()[:, :2]  # [N,2] (u,v)
        #     right_pts = veri_scaled_points_lidar2img_mis.cpu().numpy()[:, :2]  # [N,2]
        #     right_pts[:, 0] += 640 

        #     # 좌표 형상 보정
        #     left_pts = left_pts.reshape(-1, 2)  # [N,2] 보장
        #     right_pts = right_pts.reshape(-1, 2)

        #     # 좌표 범위 클리핑
        #     H, W = img_np.shape[:2]
        #     left_pts[:, 0] = np.clip(left_pts[:, 0], 0, W-1)
        #     left_pts[:, 1] = np.clip(left_pts[:, 1], 0, H-1)
        #     right_pts[:, 0] = np.clip(right_pts[:, 0], 0, W-1)
        #     right_pts[:, 1] = np.clip(right_pts[:, 1], 0, H-1)

        #     # NaN 값 필터링
        #     valid_mask = ~(np.isnan(left_pts).any(axis=1) | np.isnan(right_pts).any(axis=1))
        #     left_pts = left_pts[valid_mask]
        #     right_pts = right_pts[valid_mask]

        #     # 시각화
        #     plt.figure(figsize=(20, 6))
        #     plt.imshow(img_np)

        #     # 포인트 및 연결선 플롯
        #     plt.scatter(left_pts[:,0], left_pts[:,1], 
        #                 c='cyan', s=80, edgecolors='k', linewidths=0.8, label='Left Points')
        #     plt.scatter(right_pts[:,0], right_pts[:,1], 
        #                 c='magenta', s=80, edgecolors='k', linewidths=0.8, label='Right Points')

        #     # for left_p, right_p in zip(left_pts, right_pts):
        #     #     plt.plot([left_p[0], right_p[0]], [left_p[1], right_p[1]],
        #     #             color='yellow', linestyle='--', linewidth=1.5, alpha=0.6)

        #     plt.axis('off')
        #     plt.legend(loc='upper right', prop={'size': 12})
        #     plt.savefig('correspond.jpg', dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print ("end")
        # corrs = torch.stack(list_trimed_corrs)
        
        ########### corr transformer sjmoon ###########
        # selected_imgs, trimed_corrs ,original_camera_ids = process_queries(corrs_points,sbs_img)
        selected_imgs, trimed_corrs ,original_camera_ids = process_queries_adv(corrs_points,sbs_img)

        # # 분포 추적 버퍼 초기화 (모델 클래스 내부에 선언)
        # if not hasattr(self, 'corr_stats'):
        #     self.register_buffer('corr_mean', torch.zeros(3))
        #     self.register_buffer('corr_std', torch.ones(3))
        
        # if trimed_corrs.numel() == 0:  # 텐서가 비어있는 경우
        #     # EMA 통계 기반 샘플링 (검색 결과[1][7] 참조)
        #     # corrs_pred = torch.randn(
        #     #     (selected_imgs.size(0), 100, 3), 
        #     #     device=selected_imgs.device
        #     # ) * self.corr_std + self.corr_mean  # 핵심 변경 부분
        #     self.empty_count += 1  # 빈 텐서 발생 시 카운트 증가
        #     corrs_pred = ref_points_uvz
        #     corrs_pred[... , 0] += 0.5
            
        #     # # zero loss 생성 (requires_grad=True 유지)
        #     # corr_loss = torch.tensor(0.0, 
        #     #                     device=selected_imgs.device,
        #     #                     dtype=selected_imgs.dtype,
        #     #                     requires_grad=True)
        
        try:
            assert trimed_corrs.numel() != 0, "Empty correspondence tensor"
        except AssertionError as e:
            print(f"FATAL ERROR: {e}")
            sys.exit(1)  # 종료 코드 1 반환
        
        query_input = trimed_corrs[...,2:5]
        corr_target = trimed_corrs[...,5:]
       
        corrs_pred, cycle, corr_mask, enc_out = self.corr(selected_imgs, query_input)
        # corrs_pred = self.final_ln(corrs_pred)  # (B, N, C) 형태 유지

        # # LayerNorm 파라미터 추출
        # ln_weight = self.final_ln.weight.detach()  # (3,)
        # ln_bias = self.final_ln.bias.detach()      # (3,)

        # # corr_target 정규화 (수동 계산)
        # mean = corr_target.mean(dim=-1, keepdim=True)
        # var = corr_target.var(dim=-1, keepdim=True, unbiased=False)
        # corr_target_normalized = (corr_target - mean) / torch.sqrt(var + 1e-5)
        # corr_target_normalized = corr_target_normalized * ln_weight + ln_bias
         
        corr_loss = self.corr_loss(corrs_pred, corr_target, cycle, query_input, corr_mask)
        
        # # 그래디언트 노름 출력 코드 삽입
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad.norm()}")
        
        # 그래디언트 클리핑 적용
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)

        # self.total_iter += 1  # 모든 이터레이션에서 카운트 증가
        # # 매 100회 이터레이션마다 로깅
        # if self.total_iter % 50 == 0:
        #     empty_ratio = self.empty_count / self.total_iter
        #     print(f"[Iter {self.total_iter}] Empty count: {self.empty_count:.2%}")
        #     print(f"[Iter {self.total_iter}] Empty ratio: {empty_ratio:.2%}")

            # # 분포 통계 업데이트 (EMA 적용)
            # self.corr_mean = 0.9 * self.corr_mean + 0.1 * corrs_pred.mean(dim=(0,1))
            # self.corr_std = 0.9 * self.corr_std + 0.1 * corrs_pred.std(dim=(0,1))
         
        # ##### 검증용 display ######
        # from image_processing_unit_Ver15_0 import draw_correspondences
        # # corrs_pred_norm = self.inverse_layer_norm(corrs_pred, self.final_ln)
        # pred_corrs = torch.cat([query_input,corrs_pred],dim=2)
        # int_ids = original_camera_ids.to(torch.long).cpu()
        # for cid in int_ids :
        #     draw_correspondences(
        #         trimed_corrs=trimed_corrs[cid][:,2:],  # 첫 번째 배치 선택
        #         sbs_img=sbs_img,
        #         camera_idx=cid,
        #         save_path='correspondence_visualization_gt.jpg'
        #     )
        #     draw_correspondences(
        #         trimed_corrs=pred_corrs[cid],  # 첫 번째 배치 선택
        #         sbs_img=sbs_img,
        #         camera_idx=cid,
        #         save_path='correspondence_visualization_pred.jpg'
        #     )
        #     print ("end")

        # denormal_pred_uvz = minmax_denormalize_uvz(corrs_pred,mis_min_vals,mis_max_vals)
        denormal_pred_uvz = denormalize_points(corrs_pred)
        descale_pre_uvz = inverse_scale_uvz_points(denormal_pred_uvz)
        descale_pre_uvz_with_index = torch.cat([trimed_corrs[...,0:2],descale_pre_uvz],dim=2)
        pixel_normal_uvz = pixel_to_normalized(descale_pre_uvz_with_index,intrinsics)
        detection_xyz_adv_with_index, detection_xyz_adv ,lidar2img = center2lidar_batch(pixel_normal_uvz,intrinsics,extrinsics)
        detection_xyz_normal = detection_xyz_adv.float()

        ## query generator by SJMOON : 카메라에서 제대로 나온 라이다 xyz points 
        detection_xyz_normal[..., 0:1] = (detection_xyz_normal[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        detection_xyz_normal[..., 1:2] = (detection_xyz_normal[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        detection_xyz_normal[..., 2:3] = (detection_xyz_normal[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        detection_xyz_normal.clamp(min=0, max=1)
        pred_xyz = detection_xyz_normal.contiguous().view(-1, 3)
        trimed_pred_xyz = trim_corrs(pred_xyz,num_kp=rois.shape[0]).clone()
        # trimed_pred_xyz = self.final_ln_sparse_cross_attn(trimed_pred_xyz)

        # ####### display 용 gt uvz ############    
        # denormal_query_uvz = denormalize_points(self.inverse_layer_norm(corr_target_normalized, self.final_ln))
        # descale_query_uvz = inverse_scale_uvz_points(denormal_query_uvz)
        # descale_query_uvz_with_index = torch.cat([trimed_corrs[...,0:2],descale_query_uvz],dim=2)
        # pixel_normal_query_uvz = pixel_to_normalized(descale_query_uvz_with_index,intrinsics)
        # detection_xyz_adv_with_index, detection_query_xyz_adv ,lidar2img = center2lidar_batch(pixel_normal_query_uvz,intrinsics,extrinsics)
        # detection_query_xyz_normal = detection_query_xyz_adv.float()

        # detection_query_xyz_normal[..., 0:1] = (detection_query_xyz_normal[..., 0:1] - self.pc_range[0]) / (
        #         self.pc_range[3] - self.pc_range[0])
        # detection_query_xyz_normal[..., 1:2] = (detection_query_xyz_normal[..., 1:2] - self.pc_range[1]) / (
        #         self.pc_range[4] - self.pc_range[1])
        # detection_query_xyz_normal[..., 2:3] = (detection_query_xyz_normal[..., 2:3] - self.pc_range[2]) / (
        #         self.pc_range[5] - self.pc_range[2])
        # detection_query_xyz_normal.clamp(min=0, max=1)
        # gt_display_xyz = detection_query_xyz_normal.contiguous().view(-1, 3)
        # ########## end of display ##########
         
        # generate box correlation
        corr, mask = self.box_corr_module.gen_box_roi_correlation(rois, [len(p) for p in proposal_list], img_metas)

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
        # for i in range(6):
        #     ref_lidar_img = detection_xyz.matmul(gt_KT[i][:3, :3].T) + gt_KT[i][:3, 3].unsqueeze(0)
        #     ref_lidar_img = torch.cat([ref_lidar_img[:, :2] / ref_lidar_img[:, 2:3], ref_lidar_img[:, 2:3]], 1)
        #     depth , ref_uv,ref_z, valid_indices = points2depthmap(ref_lidar_img , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     ori_uvz = torch.cat((ref_uv, ref_z.unsqueeze(1)), dim=1)
        #     dense_depth_img_raw = dense_map_gpu_optimized(ori_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     dense_depth_img_raw = dense_depth_img_raw.to(dtype=torch.uint8).to(detection_xyz.device)
        #     dense_depth_img_color_raw = colormap(dense_depth_img_raw)

        #     ######### mis-aligned ##########
        #     ori_extrinsic = torch.from_numpy(img_metas[i]['extrinsics']).to(torch.float32).to(pts_lidar_mis.device)
        #     ori_intrinsic = torch.from_numpy(img_metas[i]['intrinsics']).to(torch.float32).to(pts_lidar_mis.device)
        #     lidar2img_original = ori_intrinsic[:3,:3] @ ori_extrinsic[:3, :] 
        #     # 1. Homogeneous 좌표로 변환
        #     pts_hom = torch.cat([pts_lidar_mis, torch.ones_like(pts_lidar_mis[:, :1])], dim=1)
        #     points_img_mis = (gt_KT[i] @ pts_hom.T).T
        #     points_img_mis = torch.cat([points_img_mis[:, :2] / points_img_mis[:, 2:3], points_img_mis[:, 2:3]], 1)
            
        #     depth ,comp_uv,comp_z, valid_indices = points2depthmap(points_img_mis , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     comp_uvz = torch.cat((comp_uv, comp_z.unsqueeze(1)), dim=1)
        #     dense_depth_img_mi_comp = dense_map_gpu_optimized(comp_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     dense_depth_img_mi_comp = dense_depth_img_mi_comp.to(dtype=torch.uint8).to(detection_xyz.device)
        #     dense_depth_img_color_mis_comp = colormap(dense_depth_img_mi_comp)

        #     # lidar_points_homo = torch.cat([detection_xyz, torch.ones_like(detection_xyz[:, :1])], dim=1)
        #     # points_img = (gt_KT[i] @ lidar_points_homo.T).T
        #     # points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)
        #     # depth , uv,z, valid_indices = points2depthmap(points_img , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     # ref_uvz = torch.cat((uv, z.unsqueeze(1)), dim=1)
        #     # dense_depth_img_ref = dense_map_gpu_optimized(ref_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     # dense_depth_img_ref = dense_depth_img_ref.to(dtype=torch.uint8).to(detection_xyz.device)
        #     # dense_depth_img_color_ref = colormap(dense_depth_img_ref)

        #     #### 예측값 디스플레이 ####
        #     denormalized_pts = pred_xyz.clone()
        #     denormalized_pts[..., 0:1] = pred_xyz[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        #     denormalized_pts[..., 1:2] = pred_xyz[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        #     denormalized_pts[..., 2:3] = pred_xyz[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        #     lidar_points_pred_homo = torch.cat([denormalized_pts, torch.ones_like(denormalized_pts[:, :1])], dim=1)
        #     points_img_pred = (gt_KT[i] @ lidar_points_pred_homo.T).T
        #     points_img_pred = torch.cat([points_img_pred[:, :2] / points_img_pred[:, 2:3], points_img_pred[:, 2:3]], 1)
        #     depth , pred_uv,pred_z, valid_indices = points2depthmap(points_img_pred , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     pred_uvz = torch.cat((pred_uv, pred_z.unsqueeze(1)), dim=1)
        #     dense_depth_img_mis = dense_map_gpu_optimized(pred_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     dense_depth_img_mis = dense_depth_img_mis.to(dtype=torch.uint8).to(detection_xyz.device)
        #     dense_depth_img_color_mis = colormap(dense_depth_img_mis)

        #     #### GT값 디스플레이 ####
        #     denormalized_pts = gt_display_xyz.clone()
        #     denormalized_pts[..., 0:1] = gt_display_xyz[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        #     denormalized_pts[..., 1:2] = gt_display_xyz[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        #     denormalized_pts[..., 2:3] = gt_display_xyz[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        #     lidar_points_pred_homo = torch.cat([denormalized_pts, torch.ones_like(denormalized_pts[:, :1])], dim=1)
        #     points_img_pred = (gt_KT[i] @ lidar_points_pred_homo.T).T
        #     points_img_pred = torch.cat([points_img_pred[:, :2] / points_img_pred[:, 2:3], points_img_pred[:, 2:3]], 1)
        #     depth , pred_uv,pred_z, valid_indices = points2depthmap(points_img_pred , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        #     pred_uvz = torch.cat((pred_uv, pred_z.unsqueeze(1)), dim=1)
        #     dense_depth_img_mis_gt = dense_map_gpu_optimized(pred_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        #     dense_depth_img_mis_gt = dense_depth_img_mis_gt.to(dtype=torch.uint8).to(detection_xyz.device)
        #     dense_depth_img_color_mis_gt = colormap(dense_depth_img_mis_gt)

        #     ###### 검증용 display########
        #     import matplotlib.pyplot as plt
        #     import matplotlib.patches as patches
        #     img_np = img[i].permute(1, 2, 0).detach().cpu().numpy()
        #     lidar_depth_mis_np = lidar_depth_mis[0].permute(1, 2, 0).detach().cpu().numpy()
        #     if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        #         img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
        #     ref_depth_np = dense_depth_img_color_raw.detach().cpu().numpy()
        #     comp_mis_depth_np = dense_depth_img_color_mis_comp.detach().cpu().numpy()
        #     mis_depth_np = dense_depth_img_color_mis.detach().cpu().numpy()
        #     depth_np = dense_depth_img_color_mis_gt.detach().cpu().numpy()

        #     # 깊이 맵의 알파 채널 설정 (투명도 조절)
        #     alpha = 0.5
        #     ref_depth_np_with_alpha = np.concatenate([ref_depth_np, np.ones((*ref_depth_np.shape[:2], 1)) * alpha], axis=2)
        #     depth_np_with_alpha = np.concatenate([depth_np, np.ones((*depth_np.shape[:2], 1)) * alpha], axis=2)
        #     mis_depth_np_with_alpha = np.concatenate([mis_depth_np, np.ones((*mis_depth_np.shape[:2], 1)) * alpha], axis=2)
        #     comp_mis_depth_np_with_alpha = np.concatenate([comp_mis_depth_np, np.ones((*comp_mis_depth_np.shape[:2], 1)) * alpha], axis=2)

        #     # 그림 생성
        #     fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(40, 20))

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

        #     ax4.imshow(img_np)
        #     # ax1.imshow(depth_np_with_alpha)
        #     ax4.imshow(depth_np_with_alpha)
        #     ax4.set_title("Object mis gt depth Map", fontsize=10)
        #     ax4.axis('off')

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
            
            all_cls_scores, all_bbox_preds = self.bbox_head(trimed_pred_xyz[:, None],
                                                            corr_feats,
                                                            ~mask[..., None, None].expand_as(corr_feats[:, :, 0]),
                                                            corr_pe,
                                                            attn_mask=None,
                                                            cross_attn_mask=None,
                                                            force_fp32=self.force_fp32, )
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)

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

        return bbox_results , corr_loss
        # return bbox_results

    # def _bbox_forward(self, x, proposal_list, img_metas): # for original 
    def _bbox_forward(self,img,img_metas,lidar_depth_mis,x,depth_x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4): ### this modified moon
        # bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas) # for original 
        bbox_results ,loss_corr = self._bbox_forward_denoise(img,img_metas,lidar_depth_mis,x, depth_x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJMOON 
        return bbox_results ,loss_corr
        # return bbox_results

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

    def forward_train(self,
                      img,
                      img_metas,
                      lidar_depth_mis,
                      x,
                      mis_depthmap_feat,
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

        mis_depth_pos_enc = self.position_encoding(mis_depthmap_feat, img_metas)
        depth_x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(mis_depthmap_feat, mis_depth_pos_enc)]

        losses = dict()

        if self.use_denoise:
            img_metas[0]['gt_bboxes_3d'] = ori_gt_bboxes_3d[0]
            img_metas[0]['gt_labels_3d'] = ori_gt_labels_3d[0]

        results_from_last,loss_corr = self._bbox_forward_train(img,img_metas,lidar_depth_mis, x, depth_x, proposal_boxes, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4) # for SJ MOON 
        # results_from_last = self._bbox_forward_train(x, proposal_boxes, img_metas) # for original 
        preds = results_from_last['pred']

        cls_scores = preds['cls_scores']
        bbox_preds = preds['bbox_preds']
        loss_weights = copy.deepcopy(self.stage_loss_weights)

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

        return losses , loss_corr
    
    def simple_test(self,img,img_metas, lidar_depth_mis, x, mis_depthmap_feat,proposal_list, uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4,rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        mis_depth_pos_enc = self.position_encoding(mis_depthmap_feat, img_metas)
        depth_x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(mis_depthmap_feat, mis_depth_pos_enc)]

        results_from_last = dict()

        results_from_last['batch_size'] = len(img_metas) // img_metas[0]['num_views']
        results_from_last ,_ = self._bbox_forward(img,img_metas, lidar_depth_mis,x,depth_x, proposal_list,uvz_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4)

        ## original
        cls_scores = results_from_last['cls_scores'][-1]
        bbox_preds = results_from_last['bbox_preds'][-1]

        # cls_scores = results_from_last[0]['cls_scores'][-1]
        # bbox_preds = results_from_last[0]['bbox_preds'][-1]

        bbox_list = self.bbox_head.get_bboxes({'cls_scores': [cls_scores], 'bbox_preds': [bbox_preds]}, img_metas,)

        return bbox_list

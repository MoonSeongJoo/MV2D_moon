# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS #,CALIB_TRANSFORMER
from .mv2d_head import MV2DHead
# from mmdet3d_plugin.models.utils import PETRTransformer
# from mmdet3d_plugin.models.utils.pe import PE

from image_processing_unit_Ver15_0 import (find_rois_nonzero_z,find_rois_nonzero_z_adv,find_rois_nonzero_z_adv1,
                                           find_rois_nonzero_z_adv2,find_rois_nonzero_z_adv3,find_rois_nonzero_z_adv3_gpu,
                                           image_to_lidar_global_modi,image_to_lidar_global_modi1,
                                           miscalib_transform, miscalib_transform1,
                                           points2depthmap,dense_map_gpu_optimized,colormap)

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
    
class CorrelationCycleLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = 1.0

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

    def forward(self, corr_pred, corr_target):
        def chamfer_distance(x, y):
            # 입력 텐서가 2차원인 경우 3차원으로 확장
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if y.dim() == 2:
                y = y.unsqueeze(0)
            xx = torch.bmm(x, x.transpose(2, 1))
            yy = torch.bmm(y, y.transpose(2, 1))
            zz = torch.bmm(x, y.transpose(2, 1))
            diag_ind = torch.arange(0, x.size(1)).to(x.device)
            rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
            ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
            P = (rx.transpose(2, 1) + ry - 2 * zz)
            P_clamp = torch.clamp(P, min=0 ,max=100)
            # return P.min(1)[0].mean() + P.min(2)[0].mean()
            return P_clamp.min(1)[0].mean() + P_clamp.min(2)[0].mean()

        chamfer_loss = chamfer_distance(corr_pred, corr_target)
        return self.loss_weight * chamfer_loss / corr_target.shape[0]

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
        self.dynamic_linear = nn.Linear(6*1*256 , 3)
        self.corr_loss = CorrelationCycleLoss()

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

    def _bbox_forward_denoise(self, img,lidar_depth_mis,x, depth_x, proposal_list,img_metas, uvz_gt,gt_KT,mis_RT): # for SJMOON
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
        # detection_nonzero_uvz = find_rois_nonzero_z(rois,points_gt)
        # detection_nonzero_uvz = find_rois_nonzero_z_adv(rois,points_gt)
        # detection_nonzero_xyz = image_to_lidar_global_modi(detection_nonzero_uvz,gt_KT) # 교정되어진 lidar좌표계 pc
        # gt_xyz = miscalib_transform(detection_nonzero_xyz,mis_RT) #mis-calibrated lidar좌표계 pc
        detection_nonzero_uvz , conf_scores = find_rois_nonzero_z_adv3(rois,uvz_gt) 
        detection_nonzero_xyz = image_to_lidar_global_modi1(detection_nonzero_uvz,gt_KT) # 교정되어진 lidar좌표계 pc
        gt_xyz = miscalib_transform1(detection_nonzero_xyz,mis_RT) #mis-calibrated lidar좌표계 pc

        # ####### input 검증용 #############
        # lidar_img = detection_nonzero_xyz[:,1:4].matmul(gt_KT[0][:3, :3].T) + gt_KT[0][:3, 3].unsqueeze(0)
        # lidar_img = torch.cat([lidar_img[:, :2] / lidar_img[:, 2:3], lidar_img[:, 2:3]], 1)
        # depth , uv,z, valid_indices = points2depthmap(lidar_img , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        # ref_uvz = torch.cat((uv, z.unsqueeze(1)), dim=1)
        # dense_depth_img_ref = dense_map_gpu_optimized(ref_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 4)
        # dense_depth_img_ref = dense_depth_img_ref.to(dtype=torch.uint8).to(dense_depth_img_ref.device)
        # dense_depth_img_color_ref = colormap(dense_depth_img_ref)

        # lidar_img_mis = gt_xyz[:,1:4].matmul(gt_KT[0][:3, :3].T) + gt_KT[0][:3, 3].unsqueeze(0)
        # lidar_img_mis = torch.cat([lidar_img_mis[:, :2] / lidar_img_mis[:, 2:3], lidar_img_mis[:, 2:3]], 1)
        # depth , gt_uv,gt_z, valid_indices = points2depthmap(lidar_img_mis , img_metas[0]['img_shape'][0] ,img_metas[0]['img_shape'][1])
        # gt_uvz = torch.cat((gt_uv, gt_z.unsqueeze(1)), dim=1)
        # dense_depth_img_mis = dense_map_gpu_optimized(gt_uvz.T , img_metas[0]['img_shape'][1] ,img_metas[0]['img_shape'][0], 1)
        # dense_depth_img_mis = dense_depth_img_mis.to(dtype=torch.uint8).to(dense_depth_img_ref.device)
        # dense_depth_img_color_mis = colormap(dense_depth_img_mis)

        # ###### 검증용 display########
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # img_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
        # lidar_depth_mis_np = lidar_depth_mis[0].permute(1, 2, 0).detach().cpu().numpy()
        # if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        #     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        # depth_np = dense_depth_img_color_ref.detach().cpu().numpy()
        # mis_depth_np = dense_depth_img_color_mis.detach().cpu().numpy()

        # # original_width, original_height = 1600, 900
        # # new_width, new_height = 1408, 512
        # # scale_x = new_width / original_width
        # # scale_y = new_height / original_height

        # # 깊이 맵의 알파 채널 설정 (투명도 조절)
        # # alpha = 0.5
        # # depth_np_with_alpha = np.concatenate([depth_np, np.ones((*depth_np.shape[:2], 1)) * alpha], axis=2)
        # # mis_depth_np_with_alpha = np.concatenate([mis_depth_np, np.ones((*mis_depth_np.shape[:2], 1)) * alpha], axis=2)

        # # # 그림 생성
        # # fig, ax = plt.subplots(figsize=(20, 10))
        # # ax.imshow(img_np)

        # # # 깊이 맵 오버레이
        # # ax.imshow(depth_np_with_alpha)

        # # # # 바운딩 박스 오버레이
        # # # for bbox in proposal_list[0]:
        # # #     x, y, w, h = bbox[:4].detach().cpu().numpy()
        # # #     new_x = int(x * scale_x)
        # # #     new_y = int(y * scale_y)
        # # #     new_w = int(w * scale_x)
        # # #     new_h = int(h * scale_y)
        # # #     rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        # # #     ax.add_patch(rect)

        # # ax.set_title("Image with Depth Map and Bounding Boxes Overlay", fontsize=15)
        # # ax.axis('off')

        # # # 전체 그림 저장
        # # plt.tight_layout()
        # # plt.savefig('verify.jpg', dpi=300, bbox_inches='tight')
        # # plt.close()

        # # 그림 생성
        # fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(40, 20))

        # # 첫 번째 서브플롯: img와 depth_np 오버레이
        # ax1.imshow(img_np)
        # # ax1.imshow(depth_np_with_alpha)
        # ax1.set_title("Image", fontsize=10)
        # ax1.axis('off')

        # # # 두 번째 서브플롯: img와 mis_depth_np 오버레이
        # ax2.imshow(depth_np)
        # # ax2.imshow(mis_depth_np_with_alpha)
        # ax2.set_title("Object depth Map", fontsize=10)
        # ax2.axis('off')

        # # # 세 번째 서브플롯: img와 mis_depth_np 오버레이
        # ax3.imshow(mis_depth_np)
        # # ax2.imshow(mis_depth_np_with_alpha)
        # ax3.set_title("Object mis depth Map", fontsize=10)
        # ax3.axis('off')

        # # 전체 그림 저장
        # plt.tight_layout()
        # plt.savefig('verify.jpg', dpi=300, bbox_inches='tight')
        # plt.close()
      
        #### query generator
        reference_points,return_feats = self.query_generator(bbox_feats, intrinsics, extrinsics, extra_feats)
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        reference_points.clamp(min=0, max=1)

        # query generator by SJMOON : 카메라에서 제대로 나온 라이다 xyz points 
        corrs_pred_normalization = detection_nonzero_xyz[:,1:].clone().detach()
        corrs_pred_normalization[..., 0:1] = (corrs_pred_normalization[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        corrs_pred_normalization[..., 1:2] = (corrs_pred_normalization[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        corrs_pred_normalization[..., 2:3] = (corrs_pred_normalization[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        corrs_pred_normalization.clamp(min=0, max=1)

        # gt_xyz normalization : 의도한대로 mis-calibrated 해서 라온 라이다 xyz points
        gt_xyz_normalization = gt_xyz[:,1:].clone().detach()
        gt_xyz_normalization[..., 0:1] = (gt_xyz_normalization[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        gt_xyz_normalization[..., 1:2] = (gt_xyz_normalization[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        gt_xyz_normalization[..., 2:3] = (gt_xyz_normalization[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        gt_xyz_normalization.clamp(min=0, max=1)
       
        # generate box correlation
        corr, mask = self.box_corr_module.gen_box_roi_correlation(rois, [len(p) for p in proposal_list], img_metas)

        depth_x_feats, dapth_pe = depth_x[0].split([ 512 // 2, 512 // 2], dim=1)

        B, C, H, W = depth_x_feats.shape
        current_tgt_len = corrs_pred_normalization.shape[0]  # 63
        conf_mask = conf_scores.view(current_tgt_len, 1, 1, 1)  # [63, 1, 1, 1]

        # [n_query, n, h, w] 형태로 expand
        conf_mask = conf_mask.expand(-1, B, H, W)  # [63, 6, 32, 88]

        # auto-calib cross attention module 
        pred_xyz_feat = self.bbox_head.forward_calib_attn(corrs_pred_normalization[None,:,:3],
                                        depth_x_feats[None], # x
                                        torch.zeros_like(depth_x_feats[None, :, 0]).bool(), #masks
                                        dapth_pe[None], # x position embedding 
                                        attn_mask=None,
                                        cross_attn_mask=conf_mask,
                                        force_fp32=self.force_fp32)
        
        # # self.weight의 크기를 확인하고 필요한 경우 조정
        # if self.weight.shape != (feature_dim, 3):
        #     self.weight = nn.Parameter(torch.Tensor(feature_dim, 3))
        #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # batch_size, seq_len, num_points, feature_dim = pred_xyz_feat.shape
        # # weight를 전치하고 확장
        # weight_expanded = self.weight.t().unsqueeze(0).expand(batch_size*seq_len, 3, feature_dim)
        # # bmm 연산 수행
        # pred_xyz = torch.bmm(pred_xyz_feat.view(-1, num_points, feature_dim), weight_expanded)
        # # 결과 텐서의 shape 조정
        # pred_xyz = pred_xyz.view(batch_size, seq_len, num_points, 3)
        
        # # 고정 layer 할당
        # batch_size, seq_len, num_points, feature_dim = pred_xyz_feat.shape
        # pred_xyz_feat = pred_xyz_feat.flatten()
        # output_size = batch_size*seq_len*num_points*feature_dim
        # output_size = pred_xyz_feat.shape
        # weight = self.weight[:output_size ,: num_points*3]
        # bias = self.bias[:num_points*3]
        # pred_xyz = F.linear(pred_xyz_feat, weight.T, bias)
        # pred_xyz = pred_xyz.view(num_points, 3)

        # # 필요한 크기로 가중치 조정
        # weight = self.weight[:, :num_points*3].T
        # bias = self.bias[:num_points*3]
        # # 메모리 효율적인 연산
        # pred_xyz = torch.bmm(pred_xyz_feat.view(-1, num_points, self.feature_dim), 
        #                      weight.unsqueeze(0).expand(batch_size*seq_len, -1, -1))
        # pred_xyz = pred_xyz + bias
        # pred_xyz = pred_xyz.view(batch_size, seq_len, num_points, 3)
        
        # # 동적 layer 할당 
        # batch_size, seq_len, num_points, feature_dim = pred_xyz_feat.shape
        # pred_xyz_feat = pred_xyz_feat.flatten()
        # dynamic_linear = nn.Linear(batch_size*seq_len*num_points*feature_dim , num_points * 3).to(pred_xyz_feat.device)
        # pred_xyz = dynamic_linear(pred_xyz_feat)
        # pred_xyz = pred_xyz.view(num_points, 3)
        
        batch_size, seq_len, num_points, feature_dim = pred_xyz_feat.shape
        pred_xyz_feat = pred_xyz_feat.view(num_points,batch_size*seq_len*feature_dim)
        pred_xyz = self.dynamic_linear(pred_xyz_feat)

        loss_corr = self.corr_loss(pred_xyz, gt_xyz_normalization[:,:3]) # arguments : corr_pred, corr_target, cycle, queries, mask

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
            
            all_cls_scores, all_bbox_preds = self.bbox_head(pred_xyz[:, None],
                                                            corr_feats,
                                                            ~mask[..., None, None].expand_as(corr_feats[:, :, 0]),
                                                            corr_pe,
                                                            attn_mask=None,
                                                            cross_attn_mask=None,
                                                            force_fp32=self.force_fp32, )

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

        return bbox_results , loss_corr

    # def _bbox_forward(self, x, proposal_list, img_metas): # for original 
    def _bbox_forward(self,img, lidar_depth_mis,x,depth_x, proposal_list,img_metas, uvz_gt,gt_KT,mis_RT): ### this modified moon
        # bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas) # for original 
        bbox_results ,loss_corr = self._bbox_forward_denoise(img,lidar_depth_mis,x, depth_x, proposal_list,img_metas,uvz_gt,gt_KT,mis_RT) # for SJMOON 
        return bbox_results ,loss_corr

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
                      lidar_depth_mis,
                      x,
                      mis_depthmap_feat,
                      proposal_list,
                      img_metas,
                      uvz_gt,
                      gt_KT,
                      mis_RT,
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

        results_from_last, loss_corr = self._bbox_forward_train(img,lidar_depth_mis, x, depth_x, proposal_boxes, img_metas, uvz_gt,gt_KT,mis_RT) # for SJ MOON 
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

        return losses ,loss_corr
    
    def simple_test(self,img,lidar_depth_mis, x, mis_depthmap_feat,proposal_list, uvz_gt,gt_KT,mis_RT,img_metas,rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        mis_depth_pos_enc = self.position_encoding(mis_depthmap_feat, img_metas)
        depth_x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(mis_depthmap_feat, mis_depth_pos_enc)]

        results_from_last = dict()

        results_from_last['batch_size'] = len(img_metas) // img_metas[0]['num_views']
        results_from_last = self._bbox_forward(img, lidar_depth_mis,x,depth_x, proposal_list,img_metas,uvz_gt,gt_KT,mis_RT,)

        # original
        # cls_scores = results_from_last['cls_scores'][-1]
        # bbox_preds = results_from_last['bbox_preds'][-1]

        cls_scores = results_from_last[0]['cls_scores'][-1]
        bbox_preds = results_from_last[0]['bbox_preds'][-1]

        bbox_list = self.bbox_head.get_bboxes({'cls_scores': [cls_scores], 'bbox_preds': [bbox_preds]}, img_metas,)

        return bbox_list

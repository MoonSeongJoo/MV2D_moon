# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp

import mmcv
import numpy as np
import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as tvtf
from torchvision.models import resnet50
import easydict
from mmcv.runner import auto_fp16

from mmdet.models.builder import DETECTORS, build_detector, build_head, build_neck #,build_calib_cross_attn
from mmdet3d.core import (bbox3d2result, box3d_multiclass_nms)
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d_plugin.models.utils.grid_mask import CustomGridMask
# from mmdet3d_plugin.datasets.pipelines.image_display import (display_depth_maps, add_calibration, points2depthmap 
#                                                              ,dense_map_gpu_optimized,add_mis_calibration)
# from COTR.COTR_models.cotr_model_moon_Ver12_0 import build
# from image_processing_unit_Ver15_0 import (two_images_side_by_side , find_depthmap_z ,find_all_depthmap_z,find_nonzero_depthmap_z,
#                                            image_to_lidar_global,display_nonzero_depthmap,trim_corrs_torch,resize_points,draw_points_torch,
#                                            normalize_point_cloud ,corrs_normalization,corrs_denormalization)

# cotr_args = easydict.EasyDict({
#                 "out_dir" : "general_config['out']",
#                 # "load_weights" : "None",
# #                 "load_weights_path" : './COTR/out/default/checkpoint.pth.tar' ,
#                 # "load_weights_path" : "./models/200_checkpoint.pth.tar",
#                 "load_weights_path" : None,
#                 "load_weights_freeze" : False ,
#                 "max_corrs" : 1000 ,
#                 "dim_feedforward" : 1024 , 
#                 "backbone" : "resnet50" ,
#                 "hidden_dim" : 312 ,
#                 "dilation" : False ,
#                 "dropout" : 0.1 ,
#                 "nheads" : 8 ,
#                 "layer" : "layer3" ,
#                 "enc_layers" : 6 ,
#                 "dec_layers" : 6 ,
#                 "position_embedding" : "lin_sine"
                
# })

# class COTR(nn.Module):
    
#     def __init__(self, num_kp=500):
#         super(COTR, self).__init__()
#         self.num_kp = num_kp
#         ##### CORR network #######
#         self.corr = build(cotr_args)
    
#     def forward(self, sbs_img , query_input):

#         for i in range(6) :
#             # multi camera batch cotr 필요
#             corrs_pred , enc_out = self.corr(sbs_img, query_input)
            
#             img_reverse_input = torch.cat([sbs_img[..., 640:], sbs_img[..., :640]], axis=-1)
#             ##cyclic loss pre-processing
#             query_reverse = corrs_pred
#             query_reverse[..., 0] = query_reverse[..., 0] - 0.5
#             cycle,_ = self.corr(img_reverse_input, query_reverse)
#             cycle[..., 0] = cycle[..., 0] - 0.5
#             mask = torch.norm(cycle - query_input, dim=-1) < 10 / 640

#         return corrs_pred , cycle , mask , enc_out

# class CorrelationCycleLoss(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         super().__init__()
#         self.loss_weight = loss_weight

#     def forward(self, corr_pred, corr_target, cycle, queries, mask):
#         corr_loss = torch.nn.functional.mse_loss(corr_pred, corr_target)
        
#         if mask.sum() > 0:
#             cycle_loss = torch.nn.functional.mse_loss(cycle[mask], queries[mask])
#             corr_loss += cycle_loss 
        
#         return self.loss_weight * corr_loss

# class ResNet50Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet = resnet50(pretrained=True)
#         self.features = nn.Sequential(*list(resnet.children())[:-2])
    
#     def forward(self, x):
#         return self.features(x)

@DETECTORS.register_module()
class MV2D(Base3DDetector):

    def __init__(self,
                 base_detector,
                 neck,
                 roi_head,
                 train_cfg=None,
                 test_cfg=None,
                 use_grid_mask=None,
                 init_cfg=None,
                 **kwargs,
                 ):
        super(Base3DDetector, self).__init__(init_cfg)

        self.base_detector = build_detector(base_detector)
        self.neck = build_neck(neck)
        if train_cfg is not None:
            roi_head.update(train_cfg=train_cfg['rcnn'])
        if test_cfg is not None:
            roi_head.update(test_cfg=test_cfg['rcnn'])
        self.roi_head = build_head(roi_head)

        self.use_grid_mask = isinstance(use_grid_mask, dict)
        if self.use_grid_mask:
            self.grid_mask = CustomGridMask(**use_grid_mask)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.num_kp= 100
        # self.corr = COTR(self.num_kp)
        # self.corr_loss = CorrelationCycleLoss(loss_weight=1.0)
        
        # self.embed_dims = 256
        # self.lidar_depth_backbone = ResNet50Backbone()
        # self.calib_cross_attn = build_calib_cross_attn(auto_calib_cross_attn)
        # self.corr_loss = CorrelationCycleLoss(loss_weight=1.0)
        # self.query_embedding = nn.Sequential(
        #         nn.Linear(self.embed_dims*3//2, self.embed_dims),
        #         nn.ReLU(),
        #         nn.Linear(self.embed_dims, self.embed_dims),
        #     )
    
    # @auto_fp16(apply_to=('img', 'points'))
    # def forward(self, return_loss=True, **kwargs):
    #     """Calls either forward_train or forward_test depending on whether
    #     return_loss=True.

    #     Note this setting will change the expected inputs. When
    #     `return_loss=True`, img and img_metas are single-nested (i.e.
    #     torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
    #     img_metas should be double nested (i.e.  list[torch.Tensor],
    #     list[list[dict]]), with the outer list indicating test time
    #     augmentations.
    #     """
    #     if return_loss:
    #         return self.forward_train(**kwargs)
    #     else:
    #         return self.forward_test(**kwargs)

    def process_2d_gt(self, gt_bboxes, gt_labels, device):
        """
        :param gt_bboxes:
            gt_bboxes: list[boxes] of size BATCH_SIZE
            boxes: [num_boxes, 4->(x1, y1, x2, y2)]
        :param gt_labels:
        :return:
        """
        return [torch.cat(
            [bboxes.to(device), torch.ones([len(labels), 1], dtype=bboxes.dtype, device=device),
             labels.unsqueeze(-1).to(bboxes.dtype)], dim=-1).to(device)
                for bboxes, labels in zip(gt_bboxes, gt_labels)]

    def process_2d_detections(self, results, device):
        """
        :param results:
            results: list[per_cls_res] of size BATCH_SIZE
            per_cls_res: list(boxes) of size NUM_CLASSES
            boxes: ndarray of shape [num_boxes, 5->(x1, y1, x2, y2, score)]
        :return:
            detections: list[ndarray of shape [num_boxes, 6->(x1, y1, x2, y2, score, label_id)]] of size len(results)
        """
        detections = [torch.cat(
            [torch.cat([torch.tensor(boxes), torch.full((len(boxes), 1), label_id, dtype=torch.float)], dim=1) for
             label_id, boxes in
             enumerate(res)], dim=0).to(device) for res in results]
        # import ipdb; ipdb.set_trace()
        if self.train_cfg is not None:
            min_bbox_size = self.train_cfg['detection_proposal'].get('min_bbox_size', 0)
        else:
            min_bbox_size = self.test_cfg['detection_proposal'].get('min_bbox_size', 0)
        if min_bbox_size > 0:
            new_detections = []
            for det in detections:
                wh = det[:, 2:4] - det[:, 0:2]
                valid = (wh >= min_bbox_size).all(dim=1)
                new_detections.append(det[valid])
            detections = new_detections

        return detections

    @staticmethod
    def box_iou(rois_a, rois_b, eps=1e-4):
        rois_a = rois_a[..., None, :]                # [*, n, 1, 4]
        rois_b = rois_b[..., None, :, :]             # [*, 1, m, 4]
        xy_start = torch.maximum(rois_a[..., 0:2], rois_b[..., 0:2])
        xy_end = torch.minimum(rois_a[..., 2:4], rois_b[..., 2:4])
        wh = torch.maximum(xy_end - xy_start, rois_a.new_tensor(0))     # [*, n, m, 2]
        intersect = wh.prod(-1)                                         # [*, n, m]
        wh_a = rois_a[..., 2:4] - rois_a[..., 0:2]      # [*, m, 1, 2]
        wh_b = rois_b[..., 2:4] - rois_b[..., 0:2]      # [*, 1, n, 2]
        area_a = wh_a.prod(-1)
        area_b = wh_b.prod(-1)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        return iou

    def complement_2d_gt(self, detections, gts, thr=0.35):
        # detections: [n, 6], gts: [m, 6]
        if len(gts) == 0:
            return detections
        if len(detections) == 0:
            return gts
        iou = self.box_iou(gts, detections)
        max_iou = iou.max(-1)[0]
        complement_ids = max_iou < thr
        min_bbox_size = self.train_cfg['detection_proposal'].get('min_bbox_size', 0)
        wh = gts[:, 2:4] - gts[:, 0:2]
        valid_ids = (wh >= min_bbox_size).all(dim=1)
        complement_gts = gts[complement_ids & valid_ids]
        return torch.cat([detections, complement_gts], dim=0)

    def extract_feat(self, img):
        return self.base_detector.extract_feat(img)

    def process_detector_feat(self, detector_feat):
        if self.with_neck:
            feat = self.neck(detector_feat)
        else:
            feat = detector_feat
        return feat
    
    # @force_fp32(apply_to=('img', 'points'))
    # def forward(self, return_loss=True, **kwargs):
    #     """Calls either forward_train or forward_test depending on whether
    #     return_loss=True.
    #     Note this setting will change the expected inputs. When
    #     `return_loss=True`, img and img_metas are single-nested (i.e.
    #     torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
    #     img_metas should be double nested (i.e.  list[torch.Tensor],
    #     list[list[dict]]), with the outer list indicating test time
    #     augmentations.
    #     """
    #     # if 'return_loss' in kwargs:
    #     #     return_loss = kwargs['return_loss']
    #     # else:
    #     #     return_loss = False  # 기본값을 False로 설정
        
    #     if return_loss:
    #         return self.forward_train(**kwargs)
    #     else:
    #         return self.forward_test(**kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      lidar_depth_gt,
                      lidar_depth_mis,
                      mis_KT,
                      mis_Rt,
                      gt_KT,
                      gt_KT_3by4,
                      gt_bboxes_2d,
                      gt_labels_2d,
                      gt_bboxes_2d_to_3d,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      attr_labels=None,
                      gt_bboxes_ignore=None):

        losses = dict()
        batch_size, num_views, c, h, w = img.shape
        img = img.view(batch_size * num_views, *img.shape[2:])
        assert batch_size == 1, 'only support batch_size 1 now'
        
        img_ori = img
        mis_KT = mis_KT.view(batch_size * num_views, *mis_KT.shape[2:])
        mis_Rt = mis_Rt.view(batch_size * num_views, *mis_Rt.shape[2:])
        gt_KT = gt_KT.view(batch_size * num_views, *gt_KT.shape[2:])
        gt_KT_3by4 = gt_KT_3by4.view(batch_size * num_views, *gt_KT_3by4.shape[2:])
    
        # lidar_depth_gt = lidar_depth_gt.view(batch_size * num_views, *lidar_depth_gt.shape[2:]).to(torch.float32) # uvz_gt
        lidar_depth_mis = lidar_depth_mis.view(batch_size * num_views, *lidar_depth_mis.shape[2:]).to(torch.float32)

        # img_resized = F.interpolate(img, size=[192, 640], mode="bilinear")
        # lidar_depth_mis_resized = F.interpolate(lidar_depth_mis, size=[h, w], mode="bilinear")

        # sbs_img = two_images_side_by_side(img_resized, lidar_depth_mis)
        # sbs_img = torch.from_numpy(sbs_img).permute(0,3,1,2).to('cuda')
        # sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        ############## input display ##########################
        # display_depth_maps(img,lidar_depth_gt,lidar_depth_mis)
        
        if self.use_grid_mask:
            img = self.grid_mask(img)
            lidar_depth_mis_resized = self.grid_mask(lidar_depth_mis)
            # img_resized = self.grid_mask(img_resized)

        # get pseudo monocular input
        # gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas:
        #   independent GT for each view
        # ori_gt_bboxes_3d, ori_gt_labels_3d:
        #   original GT for all the views
        ori_img_metas, ori_gt_bboxes_3d, ori_gt_labels_3d, ori_gt_bboxes_ignore = img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore
        gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas = [], [], [], [], [], []
        
        for i in range(batch_size):
            img_metas_views = ori_img_metas[i]

            for j in range(num_views):
                img_meta = dict(num_views=num_views)
                for k, v in img_metas_views.items():
                    if isinstance(v, list):
                        img_meta[k] = v[j]
                    elif k == 'ori_shape':
                        img_meta[k] = v[:3]
                    else:
                        img_meta[k] = v
                img_metas.append(img_meta)

            gt_labels_3d_views = ori_gt_labels_3d[i]
            gt_bboxes_3d_views = ori_gt_bboxes_3d[i].to(gt_labels_3d_views.device)
            for j in range(num_views):
                gt_ids = (gt_bboxes_2d_to_3d[i][j]).unique()
                select = gt_ids[gt_ids > -1].long()
                gt_bboxes_3d.append(gt_bboxes_3d_views[select])
                gt_labels_3d.append(gt_labels_3d_views[select])

            gt_bboxes.extend(gt_bboxes_2d[i])
            gt_labels.extend(gt_labels_2d[i])
            gt_bboxes_ignore.extend(ori_gt_bboxes_ignore[i])

        # calculate losses for 2D detector
        detector_feat = self.extract_feat(img)
        mis_depth_feat = self.extract_feat(lidar_depth_mis_resized)

        losses_detector = self.base_detector.forward_train_w_feat(
            detector_feat,
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore)
        for k, v in losses_detector.items():
            losses['det_' + k] = v

        # generate 2D detection
        self.base_detector.set_detection_cfg(self.train_cfg.get('detection_proposal'))
        with torch.no_grad():
            results = self.base_detector.simple_test_w_feat(detector_feat, img_metas)

        # process 2D detection
        detections = self.process_2d_detections(results, img.device)
        if self.train_cfg.get('complement_2d_gt', -1) > 0:
            detections_gt = self.process_2d_gt(gt_bboxes, gt_labels, img.device)
            detections = [self.complement_2d_gt(det, det_gt, thr=self.train_cfg.get('complement_2d_gt'))
                          for det, det_gt in zip(detections, detections_gt)]

        # calculate losses for 3d detector
        feat = self.process_detector_feat(detector_feat)
        mis_depthmap_feat = self.process_detector_feat(mis_depth_feat)
        # extracting mis-calibatated depthmap feature 
        # calib_attn_feat = self.lidar_depth_backbone(lidar_depth_mis)
        
        # roi_losses = self.roi_head.forward_train(feat, img_metas, detections,gt_bboxes, gt_labels,
        #                                          gt_bboxes_3d, gt_labels_3d,
        #                                          ori_gt_bboxes_3d, ori_gt_labels_3d,
        #                                          attr_labels, None)
        
        roi_losses , loss_corr = self.roi_head.forward_train(img_ori,img_metas,lidar_depth_mis, feat,mis_depthmap_feat, detections,lidar_depth_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4, gt_bboxes, gt_labels,
                                            gt_bboxes_3d, gt_labels_3d,
                                            ori_gt_bboxes_3d, ori_gt_labels_3d,
                                            attr_labels, None)
        losses['loss_corr'] = loss_corr
        losses.update(roi_losses)
        return losses

    def forward_test(self, 
                    img, 
                    img_metas,              
                    lidar_depth_mis,
                    lidar_depth_gt,
                    mis_KT,
                    mis_Rt,
                    gt_KT,
                    gt_KT_3by4,
                    **kwargs):
        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img), len(img_metas)))

        if num_augs == 1:
            return self.simple_test(img[0], img_metas[0], lidar_depth_mis[0], lidar_depth_gt[0], gt_KT[0], mis_Rt[0],mis_KT[0],gt_KT_3by4[0], **kwargs)
            # return self.simple_test(img, img_metas, lidar_depth_mis, lidar_depth_gt, gt_KT, mis_RT, **kwargs)
        else:
            return self.aug_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas,lidar_depth_mis, lidar_depth_gt, gt_KT, mis_Rt,mis_KT,gt_KT_3by4, proposal_bboxes=None, proposal_labels=None, rescale=False, **kwargs):

        # process multi-view inputs
        batch_size, num_views, c, h, w = img.shape
        img = img.view(batch_size * num_views, c, h, w)

        img_ori = img
        mis_KT = mis_KT.view(batch_size * num_views, *mis_KT.shape[2:])
        # mis_K = mis_K.view(batch_size * num_views, *mis_K.shape[2:])
        mis_Rt = mis_Rt.view(batch_size * num_views, *mis_Rt.shape[2:])
        gt_KT = gt_KT.view(batch_size * num_views, *gt_KT.shape[2:])
        gt_KT_3by4 = gt_KT_3by4.view(batch_size * num_views, *gt_KT_3by4.shape[2:])
    
        # lidar_depth_gt = lidar_depth_gt.view(batch_size * num_views, *lidar_depth_gt.shape[2:]).to(torch.float32)
        lidar_depth_mis = lidar_depth_mis.view(batch_size * num_views, *lidar_depth_mis.shape[2:]).to(torch.float32)

        lidar_depth_mis_resized = lidar_depth_mis
      
        ori_img_metas = img_metas
        img_metas = []
        gt_bboxes, gt_labels = [], []
        for i in range(batch_size):
            img_metas_views = ori_img_metas[i]
            for j in range(num_views):
                img_meta = dict(num_views=num_views)
                for k, v in img_metas_views.items():
                    if isinstance(v, list):
                        img_meta[k] = v[j]
                    elif k == 'ori_shape':
                        img_meta[k] = v[:3]
                    else:
                        img_meta[k] = v
                img_metas.append(img_meta)
            if proposal_bboxes is not None:
                gt_bboxes.extend(proposal_bboxes[i])
                gt_labels.extend(proposal_labels[i])

        detector_feat = self.extract_feat(img)
        mis_depth_feat = self.extract_feat(lidar_depth_mis_resized)

        # generate 3D detection
        self.base_detector.set_detection_cfg(self.test_cfg.get('detection_proposal'))
        det_results = self.base_detector.simple_test_w_feat(detector_feat, img_metas)
        detections = self.process_2d_detections(det_results, device=img.device)

        feat = self.process_detector_feat(detector_feat)
        mis_depthmap_feat = self.process_detector_feat(mis_depth_feat)

        # generate 3D detection
        # to -do 여기 아규먼트 수정해야 함 !! 
        bbox_outputs_all = self.roi_head.simple_test(img_ori,img_metas,lidar_depth_mis,feat,mis_depthmap_feat, detections,lidar_depth_gt,mis_KT,mis_Rt,gt_KT,gt_KT_3by4,rescale=rescale)
        bbox_outputs = []
        box_type_3d = img_metas[0]['box_type_3d']

        # 3D NMS
        for i in range(batch_size):
            # bbox_outputs_i: len(num_views)
            bbox_outputs_i = bbox_outputs_all[i * num_views:i * num_views + num_views]
            all_bboxes = box_type_3d.cat([x[0] for x in bbox_outputs_i])
            all_scores = torch.cat([x[1] for x in bbox_outputs_i])
            all_classes = torch.cat([x[2] for x in bbox_outputs_i])

            all_scores_classes = all_scores.new_zeros(
                (len(all_scores), self.roi_head.num_classes + 1)).scatter_(1, all_classes[:, None], all_scores[:, None])

            cfg = self.test_cfg.get('rcnn')
            results = box3d_multiclass_nms(all_bboxes.tensor, all_bboxes.bev,
                                           all_scores_classes, cfg.score_thr, cfg.max_per_scene, cfg.nms)

            bbox_outputs.append((
                box_type_3d(results[0], box_dim=all_bboxes.tensor.shape[1], with_yaw=all_bboxes.with_yaw),
                results[1], results[2]))

        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_outputs
        ]

        bbox_list = [dict() for i in range(batch_size)]
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
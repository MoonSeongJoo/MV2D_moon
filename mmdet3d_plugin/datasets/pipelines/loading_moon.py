# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.loading import LoadAnnotations3D
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F

from mmdet3d_plugin.datasets.pipelines.image_display import (points2depthmap_cpu,points2depthmap_gpu,
                                                             add_calibration,add_calibration_adv,add_mis_calibration_ori,add_calibration_adv2,
                                                             add_mis_calibration,add_mis_calibration_cpu,add_mis_calibration_adv,
                                                             dense_map_gpu_optimized,distance_adaptive_depth_completion,
                                                             colormap,preprocess_points,edge_aware_bilateral_filter,
                                                             visualize_depth_maps,
                                                             enhanced_geometric_propagation,direction_aware_completion,direction_aware_bilateral_filter,
                                                             )

@PIPELINES.register_module()
class LoadAnnotationsMono3D(LoadAnnotations3D):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(LoadAnnotationsMono3D, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def _load_bboxes_2d(self, results):
        results['gt_bboxes_2d'] = results['ann_info']['gt_bboxes_2d']
        results['gt_labels_2d'] = results['ann_info']['gt_labels_2d']
        results['gt_bboxes_2d_to_3d'] = results['ann_info']['gt_bboxes_2d_to_3d']
        results['gt_bboxes_ignore'] = results['ann_info']['gt_bboxes_ignore']
        results['bbox2d_fields'].append('gt_bboxes_2d')
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_bbox_2d:
            results = self._load_bboxes_2d(results)
        return results
    

@PIPELINES.register_module()
class LoadMapsFromFiles(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(256,3*256)
        results['map_shape']=maps.shape
        results['maps']=maps
        return results
    
@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_moon(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        # # 수정된 코드 (float32 변환 추가)
        # results['img_ori'] = [arr.astype(np.float32) for arr in results['img']]

        # ############ input image display ###############
        # plt.figure(figsize=(20, 20))
        # plt.subplot(1,1,1)

        # # Get the image data
        # img_to_display = results['img_ori'][0].copy()  # Make a copy to avoid modifying original
        # img_to_display = img_to_display[:, :, ::-1] 

        # # Print debug info (optional)
        # print(f"Image shape: {img_to_display.shape}, dtype: {img_to_display.dtype}")
        # print(f"Image min: {img_to_display.min()}, max: {img_to_display.max()}")

        # # Properly normalize the float32 image for display
        # if img_to_display.max() > 1.0:
        #     # Likely in range 0-255, normalize to 0-1
        #     img_to_display = img_to_display / 255.0
        # else:
        #     # If already in 0-1 range or has negative values, normalize properly
        #     img_to_display = (img_to_display - img_to_display.min()) / (img_to_display.max() - img_to_display.min() + 1e-8)

        # # Ensure values are clipped to valid range
        # img_to_display = np.clip(img_to_display, 0, 1)

        # # Use explicit parameters for reliable display
        # plt.imshow(img_to_display)
        # plt.title("image only", fontsize=10)

        # plt.tight_layout()
        # plt.savefig('load_pipeline_image_only.jpg', dpi=300, bbox_inches='tight')
        # plt.close()
        # print("end of print")
        # ###################################################
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
    
    # def __call__(self, results):
    #     filenames = results['img_filename']
    #     valid_images = []
    #     valid_indices = []
        
    #     for idx, name in enumerate(filenames):
    #         try:
    #             img = mmcv.imread(name, self.color_type)
    #             if img is None:
    #                 if valid_images:  # 이전 유효한 이미지가 있는 경우
    #                     img = valid_images[-1].copy()  # 마지막 유효한 이미지를 복사
    #                     print(f"Image {name} is corrupted. Using the previous valid image.")
    #                 else:
    #                     raise ValueError(f"Image {name} is corrupted and no previous valid image exists.")
    #             valid_images.append(img)
    #             valid_indices.append(idx)
    #         except Exception as e:
    #             print(f"Skipping file {name} due to error: {e}")
    #             if valid_images:
    #                 valid_images.append(valid_images[-1].copy())
    #                 valid_indices.append(idx)
    #             continue
        
    #     if not valid_images:
    #         raise ValueError("No valid images found.")
        
    #     # Update all relevant data in results
    #     for key in results.keys():
    #         if isinstance(results[key], list) and len(results[key]) == len(filenames):
    #             results[key] = [results[key][i] if i in valid_indices else results[key][valid_indices[-1]] for i in range(len(filenames))]
        
    #     # img is of shape (h, w, c, num_views)
    #     img = np.stack(valid_images, axis=-1)
    #     if self.to_float32:
    #         img = img.astype(np.float32)
        
    #     results['img_filename'] = filenames  # 원래 파일 이름 유지
    #     results['img'] = [img[..., i] for i in range(img.shape[-1])]
    #     results['img_shape'] = img.shape
    #     results['ori_shape'] = img.shape
    #     results['pad_shape'] = img.shape
    #     results['scale_factor'] = 1.0
    #     num_channels = 1 if len(img.shape) < 3 else img.shape[2]
    #     results['img_norm_cfg'] = dict(
    #         mean=np.zeros(num_channels, dtype=np.float32),
    #         std=np.ones(num_channels, dtype=np.float32),
    #         to_rgb=False)
        
    #     return results



@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                ):

        self.sweeps_num = sweeps_num    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                # import ipdb; ipdb.set_trace()
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list  

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"

import cv2

class DepthCompletionPipeline:
    def __init__(self, max_iter=100, tol=1e-5):
        # Motion estimation parameters
        self.max_iter = max_iter
        self.tol = tol
        self.prev_features = None
        self.prev_points = None
        self.orb = cv2.ORB_create(nfeatures=1000)
    
    def extract_features(self, img_tensor):
        """이미지에서 ORB 특징점 추출"""
        # 차원 재구성 [H,W,C] → [C,H,W]
        if img_tensor.shape[-1] == 3:  # 채널이 마지막 차원인 경우
            img_tensor = img_tensor.permute(2, 0, 1)  # [C,H,W]
        
        # 1/2 해상도로 다운샘플링
        small_img = F.avg_pool2d(img_tensor.unsqueeze(0), 2).squeeze(0)  # [C,H/2,W/2]
        
        # 그레이스케일 변환
        gray = 0.299 * small_img[0] + 0.587 * small_img[1] + 0.114 * small_img[2]
        
        # 정규화 및 타입 변환 (0~255 uint8)
        gray = (gray * 255).clamp(0, 255).to(torch.uint8)
        
        # CPU로 이동 및 차원 압축
        gray_np = gray.cpu().squeeze().numpy()  # [H/2,W/2]
        
        # 특징점 검출
        fast = cv2.FastFeatureDetector_create(threshold=50)
        keypoints = fast.detect(gray_np, None)
        
        # 원본 해상도 좌표 복원 (2배 스케일링)
        kp_coords = torch.tensor(
            [[kp.pt[0]*2, kp.pt[1]*2] for kp in keypoints],
            dtype=torch.float32,
            device=img_tensor.device
        )
        
        return kp_coords

    def motion_based_calibration(self, current_features, current_points):
        """
        움직임 기반 미스캘리브레이션 보정
        """
        if self.prev_features is None:
            self.prev_features = current_features
            self.prev_points = current_points
            return None

        # 특징점 기반 상대 운동 추정
        transform, _ = cv2.estimateAffinePartial2D(
            self.prev_features.cpu().numpy(),
            current_features.cpu().numpy()
        )
        
        # LiDAR 포인트 기반 운동 추정
        lidar_transform = self.estimate_relative_motion(
            self.prev_points, current_points
        )

        # 캘리브레이션 오차 계산
        calibration_error = self.calculate_calibration_error(
            transform, lidar_transform
        )

        # 보정 행렬 계산
        correction_matrix = self.compute_correction_matrix(calibration_error)
        
        self.prev_features = current_features
        self.prev_points = current_points
        
        return correction_matrix

    def depth_completion(self, sparse_depth, rgb_image, superpixel_size=100):
        """
        결정론적 가이드 depth completion
        """
        # RGB 이미지 과분할
        superpixels = self.slic_segmentation(rgb_image, superpixel_size)
        
        # 슈퍼픽셀 평면 근사화
        planar_depth = self.approximate_planar_surfaces(
            sparse_depth, superpixels
        )
        
        # Joint Bilateral Upsampling
        dense_depth = self.joint_bilateral_upsampling(
            planar_depth, rgb_image
        )
        
        return dense_depth

    def post_processing(self, depth_map, kernel_size=5):
        """
        미스얼라인먼트 강건 필터링
        """
        # Median 필터링으로 아티팩트 제거
        filtered_depth = cv2.medianBlur(depth_map.numpy(), kernel_size)
        return torch.from_numpy(filtered_depth)

    # Helper methods ----------------------------------------------------------
    def slic_segmentation(self, image, n_segments=100):
        from skimage.segmentation import slic
        
        # 차원 재구성 (H,W,C) → (C,H,W) → (1,C,H,W)
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)  # [1,3,512,1408]
        
        # 1/4 해상도 처리 (H:512→128, W:1408→352)
        small_img = F.avg_pool2d(image.float(), kernel_size=4, stride=4)  # [1,3,128,352]
        
        # NumPy 변환 (H,W,C) 형태로 조정
        small_img_np = small_img.squeeze(0).permute(1,2,0).cpu().numpy()  # [128,352,3]
        
        # SLIC 분할
        segments = slic(small_img_np, 
                    n_segments=n_segments//4,
                    compactness=10,
                    sigma=1)
        
        # 텐서 변환 및 고해상도 복원
        segments_tensor = torch.from_numpy(segments).float().to(image.device)
        segments_upsampled = F.interpolate(segments_tensor.unsqueeze(0).unsqueeze(0), 
                                        scale_factor=4, 
                                        mode='nearest').squeeze()  # [512,1408]
        
        return segments_upsampled

    def approximate_planar_surfaces(self, depth, superpixels):
        planar_depth = torch.zeros_like(depth)
        h, w = depth.shape
        
        yy, xx = torch.meshgrid(torch.arange(h, device=depth.device), 
                            torch.arange(w, device=depth.device),
                            indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).float()
        
        processed_segments = 0
        
        for seg_id in torch.unique(superpixels):
            mask = (superpixels == seg_id)
            valid_coords = coords[mask]
            z_values = depth[mask]
            
            # 유효 포인트 필터링
            valid_mask = z_values > 0
            if valid_mask.sum() < 3:
                planar_depth[mask] = 0
                continue
                
            valid_coords = valid_coords[valid_mask]
            z_values = z_values[valid_mask]
            
            # 좌표 정규화
            mean_xy = valid_coords.mean(dim=0)
            mean_z = z_values.mean()
            std_xy = valid_coords.std(dim=0) + 1e-6
            std_z = z_values.std() + 1e-6
            
            points_norm = torch.cat([
                (valid_coords - mean_xy)/std_xy,
                (z_values - mean_z).unsqueeze(-1)/std_z
            ], dim=1)
            
            # RANSAC 평면 적합
            plane_norm = self.ransac_plane_fitting(points_norm)
            if plane_norm is None:
                continue
                
            # 좌표계 변환
            a_p, b_p, c_p, d_p = plane_norm
            A = a_p * std_xy[1] * std_z
            B = b_p * std_xy[0] * std_z
            C = c_p * std_xy[0] * std_xy[1]
            D = -a_p*std_xy[1]*std_z*mean_xy[0] - b_p*std_xy[0]*std_z*mean_xy[1] - c_p*std_xy[0]*std_xy[1]*mean_z + d_p*std_xy[0]*std_xy[1]*std_z
            
            # 깊이 계산
            x = coords[mask][:, 0]
            y = coords[mask][:, 1]
            planar_z = (-A*x - B*y - D) / (C + 1e-6)
            planar_z = torch.clamp(planar_z, min=0.0)
            
            # MAD 기반 클램핑
            z_median = torch.median(planar_z)
            z_mad = 1.4826 * torch.median(torch.abs(planar_z - z_median))
            planar_z = torch.clamp(planar_z, 
                                max=z_median + 3*z_mad, 
                                min=z_median - 3*z_mad)
            
            planar_depth[mask] = planar_z
            processed_segments += 1

        print(f"처리된 세그먼트 수: {processed_segments}")
        print(f"최종 깊이 범위: {torch.min(planar_depth)} ~ {torch.max(planar_depth)}")
        return planar_depth

    def _core_jbu(self, low_res, guide, kernel_size, sigma_range):
        """JBU 핵심 연산 (저해상도 입력 전용)"""
        # 입력 차원 검증
        assert low_res.dim() == 2 and guide.dim() == 3, "Input must be [H,W] and [C,H,W]"
        
        # 가이드 이미지 그레이스케일 변환
        guide_gray = 0.299*guide[0] + 0.587*guide[1] + 0.114*guide[2]
        
        # 패딩 계산
        pad = kernel_size // 2
        h, w = low_res.shape
        
        # 공간 커널 생성
        grid = torch.arange(-pad, pad+1, device=low_res.device)
        y, x = torch.meshgrid(grid, grid)
        spatial_kernel = torch.exp(-(x**2 + y**2)/(2*(kernel_size/3)**2))
        
        # Unfold 연산
        guide_unfold = F.unfold(guide_gray.unsqueeze(0).unsqueeze(0), 
                            kernel_size=kernel_size, 
                            padding=pad).squeeze(0)  # [k*k, H*W]
        
        low_res_unfold = F.unfold(low_res.unsqueeze(0).unsqueeze(0), 
                                kernel_size=kernel_size, 
                                padding=pad).squeeze(0)  # [k*k, H*W]
        
        # 범위 커널 계산
        center_idx = (kernel_size**2) // 2
        center_values = guide_unfold[center_idx:center_idx+1]  # [1, H*W]
        range_kernel = torch.exp(-(guide_unfold - center_values)**2/(2*sigma_range**2))
        
        # 결합 가중치 계산
        combined_weights = spatial_kernel.view(-1,1) * range_kernel  # [k*k, H*W]
        norm = combined_weights.sum(dim=0) + 1e-8
        
        # 가중 평균 계산
        weighted_sum = (low_res_unfold * combined_weights).sum(dim=0)
        output = weighted_sum / norm
        
        return output.view(h, w)  # [H,W]

    def joint_bilateral_upsampling(self, low_res, guide, sigma_spatial=5, sigma_range=0.1):
        """최적화된 JBU (1/2 해상도 처리 + 고속 복원)"""
        # 차원 보정
        if low_res.dim() == 2:
            low_res = low_res.unsqueeze(0)  # [1,H,W]
        if guide.dim() == 3 and guide.shape[0] == 3:
            guide = guide.permute(1,2,0)    # [H,W,C]
        
        # 1/2 해상도 다운샘플링
        low_res_small = F.avg_pool2d(low_res, 2).squeeze(0)  # [H/2,W/2]
        guide_small = F.avg_pool2d(guide.permute(2,0,1), 2).permute(1,2,0)  # [H/2,W/2,C]
        
        # 저해상도 JBU 실행
        kernel_size = int(3*sigma_spatial)
        output_small = self._core_jbu(low_res_small, 
                                    guide_small.permute(2,0,1), 
                                    kernel_size, 
                                    sigma_range)
        
        # 고해상도 복원
        return F.interpolate(output_small.unsqueeze(0).unsqueeze(0), 
                        scale_factor=2, 
                        mode='bilinear').squeeze()
    
    def ransac_plane_fitting(self, points, max_iters=100, threshold=0.3):
        best_plane = None
        best_inliers = 0
        best_dists = None
        
        for _ in range(max_iters):
            # 3개 포인트 샘플링
            sample_indices = torch.randperm(len(points))[:3]
            p1, p2, p3 = points[sample_indices]
            
            # 평면 방정식 계산
            v1 = p2 - p1
            v2 = p3 - p1
            normal = torch.cross(v1, v2)
            norm = torch.norm(normal)
            if norm < 1e-6: continue
            
            normal /= norm
            d = -torch.dot(normal, p1)
            
            # 인라이어 계산
            dists = torch.abs(points @ normal + d)
            inliers = torch.sum(dists < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = torch.tensor([*normal, d], device=points.device)
                best_dists = dists  # dists 저장
        
        # 최종 인라이어 포인트 재계산
        if best_plane is not None:
            final_dists = torch.abs(points @ best_plane[:3] + best_plane[3])
            inlier_mask = final_dists < threshold
            return self.fit_plane_svd(points[inlier_mask])
        return best_plane
        
    def fit_plane_svd(self, points):
        """
        SVD를 사용하여 포인트 클라우드에 가장 잘 맞는 평면을 찾습니다.
        
        Args:
            points (Tensor): 포인트 클라우드 [N, 3]
            
        Returns:
            Tensor: 평면 방정식 계수 [a, b, c, d], ax + by + cz + d = 0
        """
        # 중심점 계산
        centroid = torch.mean(points, dim=0)
        
        # 중심으로 이동
        centered_points = points - centroid
        
        # 공분산 행렬 계산
        cov = torch.matmul(centered_points.T, centered_points)
        
        # SVD 수행
        try:
            U, S, Vh = torch.linalg.svd(cov)
            # 가장 작은 고유값에 해당하는 고유벡터가 법선 벡터
            normal = U[:, 2]
        except:
            # SVD 실패 시 Z축 방향 법선 사용
            normal = torch.tensor([0., 0., 1.], device=points.device)
        
        # 평면 방정식 계수
        a, b, c = normal
        d = -torch.dot(normal, centroid)
        
        return torch.tensor([a, b, c, d], device=points.device)


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config
        self.num_points =900
        self.grid_size = 30
    
    def __call__(self, results):
        raw_points_lidar = results['points']
        # points_lidar = image_display.trim_corrs(raw_points_lidar)
        points_lidar = raw_points_lidar.tensor[:, :3].clone().to(dtype=torch.float32)

        point2img_gt =[]
        lidar_depth_map_mis=[]
        lidar_depth_map_gt =[]
        list_gt_KT ,list_mis_RT ,list_mis_KT = [] ,[] ,[]
        list_gt_KT_3by4 =[]
        img_ori =[]
        for cid in range(len(results['lidar2img'])):
            lidar2img = torch.from_numpy(results['lidar2img'][cid]).to(dtype=torch.float32)
            lidar2cam = torch.from_numpy(results['extrinsics'][cid]).to(dtype=torch.float32)
            cam2img = torch.from_numpy(results['intrinsics'][cid]).to(dtype=torch.float32)
            
            # raw_img_np = results['img'][cid]
            # img_bgrtorgb = raw_img_np[:, :, ::-1].copy()

            # aug_img_np = results['img'][cid]
            # aug_img_bgrtorgb = aug_img_np[:, :, ::-1].copy()
                         
            img_height = results['img'][0].shape[0]
            img_width  = results['img'][0].shape[1]

            # img_height_aug = results['img'][0].shape[0]
            # img_width_aug  = results['img'][0].shape[1]
            # resized_img = cv2.resize(
            #     img_bgrtorgb, 
            #     (img_width, img_height),  # (width, height) 순서
            #     interpolation=cv2.INTER_LINEAR  # 기본값
            # )
            # img_to_display = img_bgrtorgb / 255.0
            # aug_img_to_display = aug_img_bgrtorgb / 255.0
            
            # img = torch.from_numpy(img_to_display).to(dtype=torch.float32)
            # aug_img = torch.from_numpy(aug_img_to_display).to(dtype=torch.float32)
           
            # points2img = add_calibration_adv(lidar2img , points_lidar)
            points2img , KT_ori  = add_calibration_adv2(lidar2cam ,cam2img, points_lidar)
            # miscalibrated_points2img_ori , mis_RT_ori , mis_KT_ori ,mis_K_ori = add_mis_calibration_ori(lidar2cam,cam2img, points_lidar,max_r=0.0,max_t=0.0)
            miscalibrated_points2img , extrinsic_perturb, lidar2img_original ,lidar2img_mis = add_mis_calibration_adv(lidar2img,lidar2cam,cam2img, points_lidar, max_r=10.0,max_t=0.075)

            point2img_gt.append(points2img) # lidar coordination 3d
            list_mis_RT.append(extrinsic_perturb) # lidar coordination 3d mis-calibration
            list_gt_KT.append(lidar2img)
            list_gt_KT_3by4.append(lidar2img_original)
            list_mis_KT.append(lidar2img_mis)
            # img_ori.append(img)
            
            ####### depth image display ######
            depth_gt, gt_uv,gt_z, valid_indices_gt= points2depthmap_gpu(points2img, img_height ,img_width)
            # lidarOnImage_gt = torch.cat((gt_uv, gt_z.unsqueeze(1)), dim=1)
            # pts = lidarOnImage_gt.T
            # dense_depth_img_gt = dense_map_gpu_optimized(pts , img_width, img_height, 4)
            # dense_depth_img_gt = dense_depth_img_gt.to(dtype=torch.uint8)
            # dense_depth_img_color_gt = colormap(dense_depth_img_gt)

            depth_mis, uv,z,valid_indices = points2depthmap_gpu(miscalibrated_points2img, img_height ,img_width)
            # lidarOnImage_mis = torch.cat((uv, z.unsqueeze(1)), dim=1)
            # # pts = preprocess_points(lidarOnImage_mis.T)
            # pts_mis = lidarOnImage_mis.T
            # dense_depth_img_mis = dense_map_gpu_optimized(pts_mis , img_width, img_height, 4)
            # # dense_depth_img_mis = distance_adaptive_depth_completion(pts , results['img'][0].shape[1], results['img'][0].shape[0], 4)
            # dense_depth_img_mis = dense_depth_img_mis.to(dtype=torch.uint8)
            # dense_depth_img_color_mis = colormap(dense_depth_img_mis)
            # dense_depth_img_edge_mis = edge_aware_bilateral_filter(pts,dense_depth_img_color_mis_raw,results['img'][0].shape[1], results['img'][0].shape[0], 4)
            # dense_depth_img_edge_mis = dense_depth_img_edge_mis.to(dtype=torch.uint8)
            # dense_depth_img_color_mis = colormap(dense_depth_img_edge_mis)

            # # 미스캘리브레이션 보정
            # calibration_pipeline = DepthCompletionPipeline()
            # features = calibration_pipeline.extract_features(img)
            # correction_matrix = calibration_pipeline.motion_based_calibration(
            #     features, points_lidar[:,:3]
            # )
            
            # if correction_matrix is not None:
            #     # LiDAR 데이터 보정
            #     points_lidar = self.apply_correction(
            #         points_lidar, correction_matrix
            #     )
            
            # # Depth Completion
            # dense_depth = calibration_pipeline.depth_completion(
            #     depth_mis, 
            #     img
            # )
            
            # # 후처리
            # final_depth = calibration_pipeline.post_processing(dense_depth)

            # # dense_depth_img_mis_adv = enhanced_geometric_propagation(depth_mis)
            # dense_depth_img_mis_adv = final_depth.to(dtype=torch.uint8)
            # dense_depth_img_color_mis_adv = colormap(dense_depth_img_mis_adv)

            # lidar_depth_dense_gt.append(dense_depth_img_color_gt)
            lidar_depth_map_mis.append(depth_mis)
            # gt_uvz.append(dense_depth_img_gt)
            lidar_depth_map_gt.append(depth_gt)
            # uvz.append(dense_depth_img_mis)
            
            # ###### input display ######
            # img_np = img.detach().cpu().numpy()
            # aug_img_np = aug_img.detach().cpu().numpy()
            # # 이미지 데이터가 float 타입인 경우 0과 1 사이로 정규화
            # # if img.dtype == np.float32 or img.dtype == np.float64:
            # #     img = (img - img.min()) / (img.max() - img.min())
            # plt.figure(figsize=(20, 20))
            # plt.subplot(4,1,1)
            # plt.imshow(img_np)
            # plt.scatter(gt_uv[:, 0], gt_uv[:, 1], c=gt_z, s=0.5)
            # plt.title("input calibrated display", fontsize=10)

            # plt.subplot(4,1,2)
            # plt.imshow(aug_img_np)
            # plt.scatter(uv[:, 0], uv[:, 1], c=z, s=0.5)
            # plt.title("input mis-calibrated display", fontsize=10)

            # disp_gt2 = dense_depth_img_color_gt.detach().cpu().numpy()
            # plt.subplot(4,1,3)
            # plt.imshow(disp_gt2, cmap='magma')
            # plt.title("gt display", fontsize=10)
            # plt.axis('off')

            # disp_mis2 = dense_depth_img_color_mis.detach().cpu().numpy()
            # plt.subplot(4,1,4)
            # plt.imshow(disp_mis2, cmap='magma')
            # plt.title("mis display", fontsize=10)
            # plt.axis('off')

            # # gt_gray = dense_depth_img_color_mis_adv.detach().cpu().numpy()
            # # plt.subplot(5,1,5)
            # # plt.imshow(gt_gray, cmap='magma_r')
            # # plt.title("other mis display", fontsize=10)
            # # plt.axis('off')

            # # # mis_gray = dense_depth_img_mis.detach().cpu().numpy()
            # # # plt.subplot(3,2,6)
            # # # plt.imshow(mis_gray, cmap='magma_r')
            # # # plt.title("mis gray display", fontsize=10)
            # # # plt.axis('off')
            
            # # 전체 그림 저장
            # plt.tight_layout()
            # plt.savefig('load_pipeline.jpg', dpi=300, bbox_inches='tight')
            # plt.close()
            # print ("end of print")
        
        # img_original = torch.stack(img_ori)
        gt_KT = torch.stack(list_gt_KT)
        gt_KT_3by4 = torch.stack(list_gt_KT_3by4)
        mis_RT = torch.stack(list_mis_RT)
        mis_KT = torch.stack(list_mis_KT)
        lidar_depth_mis = torch.stack(lidar_depth_map_mis)
        lidar_depth_gt = torch.stack(lidar_depth_map_gt)
        # lidar_depth_mis = lidar_depth_mis.permute(0, 3, 1, 2)
        # img_original = img_original.permute(0,3,1,2)
        # lidar_depth_gt = F.interpolate(lidar_depth_gt, size=[192, 640], mode="bilinear") # lidar 2d depth map input [192,640,1]
        # lidar_depth_mis = F.interpolate(lidar_depth_mis, size=[192, 640], mode="bilinear") 
        # img_original = F.interpolate(img_original, size=[img_height_aug, img_width_aug], mode="bilinear") 

        # results['img_original']  = img_original
        results['lidar_depth_gt']  = lidar_depth_gt
        results['lidar_depth_mis'] = lidar_depth_mis
        results['mis_KT'] = mis_KT
        results['mis_Rt'] = mis_RT
        results['gt_KT'] = gt_KT
        results['gt_KT_3by4'] = gt_KT_3by4

        return results
    
    def apply_correction(self, points, matrix):
        """보정 행렬 적용"""
        homog_points = torch.cat([points[:,:3], 
                                torch.ones(points.size(0),1)], dim=1)
        corrected = homog_points @ matrix.T
        return torch.cat([corrected[:,:3], points[:,3:]], dim=1)
    
    # def __call__(self, results):
    #     # CPU에서 처리
    #     raw_points_lidar = results['points'].tensor.numpy()  # GPU 텐서를 NumPy 배열로 변환
    #     points_lidar = raw_points_lidar
        
    #     num_cameras = len(results['lidar2img'])
        
    #     # NumPy 배열로 초기화
    #     point2img_gt = np.zeros((num_cameras, points_lidar.shape[0], 3))  # (num_cam, N, 3)
    #     point2img_mis = np.zeros_like(point2img_gt)
    #     lidar_depth_map_mis = np.zeros((num_cameras, *results['img'][0].shape[:2], 3), dtype=np.uint8)
    #     # lidar_depth_map_mis = np.zeros((num_cameras, *results['img'][0].shape[:2]))
    #     lidar_depth_map_gt  = np.zeros((num_cameras, *results['img'][0].shape[:2]))
    #     list_gt_KT = np.zeros((num_cameras, 4, 4))
    #     list_mis_RT = np.zeros_like(list_gt_KT)
        
    #     # NumPy 배열로 변환
    #     lidar2img = np.stack([results['lidar2img'][i] for i in range(num_cameras)])
    #     lidar2cam = np.stack([results['extrinsics'][i] for i in range(num_cameras)])
    #     cam2img = np.stack([results['intrinsics'][i] for i in range(num_cameras)])
        
    #     for cid in range(num_cameras):
    #         # CPU에서 연산 수행
    #         points2img = add_calibration_cpu(lidar2img[cid], points_lidar)
    #         miscalibrated_points2img, mis_RT = add_mis_calibration_cpu(lidar2cam[cid], cam2img[cid], points_lidar, max_r=10., max_t=0.075)
            
    #         point2img_gt[cid] = points2img
    #         point2img_mis[cid] = miscalibrated_points2img
    #         list_gt_KT[cid] = lidar2img[cid]
    #         list_mis_RT[cid] = mis_RT
            
    #         depth_gt, _, _, _ = points2depthmap_cpu(points2img, results['img'][0].shape[0], results['img'][0].shape[1])
    #         depth_mis, uv, z, _ = points2depthmap_cpu(miscalibrated_points2img, results['img'][0].shape[0], results['img'][0].shape[1])
            
    #         lidarOnImage_mis = np.concatenate((uv, z[:, np.newaxis]), axis=1)
    #         dense_depth_img_mis = dense_map_cpu_optimized(lidarOnImage_mis.T, results['img'][0].shape[1], results['img'][0].shape[0], 4)
    #         dense_depth_img_mis = dense_depth_img_mis.astype(np.uint8)
    #         dense_depth_img_color_mis = colormap(dense_depth_img_mis)
            
    #         lidar_depth_map_mis[cid] = dense_depth_img_color_mis
    #         # lidar_depth_map_mis[cid] = depth_mis
    #         # lidar_depth_map_mis[cid] = np.stack([depth_mis, depth_mis, depth_mis], axis=-1)
    #         lidar_depth_map_gt[cid] = depth_gt
        
    #     # 결과 저장 (NumPy 배열로 저장)
    #     results['lidar_depth_gt'] = lidar_depth_map_gt
    #     # results['lidar_depth_mis'] = lidar_depth_map_mis
    #     # 수정 후 (CHW 형식으로 변환)
    #     results['lidar_depth_mis'] = np.transpose(lidar_depth_map_mis, (0, 3, 1, 2))
    #     # results['reduce_points_raw'] = points_lidar[:, :3]
    #     results['mis_RT'] = list_mis_RT
    #     results['gt_KT'] = list_gt_KT

    #     # 결과 저장 후 시각화
    #     visualize_depth_maps(results['lidar_depth_gt'], results['lidar_depth_mis'])
        
    #     return results

@PIPELINES.register_module()
class ToPytorchTensor(object):
    """Convert ndarrays in sample to Tensors while preserving integer types."""
    def __call__(self, results):
        tensor_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if value.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                    # 정수형 타입 유지
                    tensor_results[key] = torch.from_numpy(value).long()
                else:
                    # 부동소수점 타입은 float로 변환
                    tensor_results[key] = torch.from_numpy(value).float()
            else:
                tensor_results[key] = value
        return tensor_results



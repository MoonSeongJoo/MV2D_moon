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
                                                             dense_map_gpu_optimized,dense_map_cpu_optimized,
                                                             colormap,colormap_cpu,
                                                             visualize_depth_maps)

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
        points_lidar = raw_points_lidar.clone()

        point2img_gt =[]
        lidar_depth_map_mis=[]
        lidar_depth_map_gt =[]
        list_gt_KT ,list_mis_RT ,list_mis_KT = [] ,[] ,[]
        list_gt_KT_3by4 =[]
        for cid in range(len(results['lidar2img'])):
            lidar2img = torch.from_numpy(results['lidar2img'][cid]).to(torch.float32)
            lidar2cam = torch.from_numpy(results['extrinsics'][cid]).to(torch.float32)
            cam2img = torch.from_numpy(results['intrinsics'][cid]).to(torch.float32)

            # points2img = add_calibration_adv(lidar2img , points_lidar)
            points2img , KT_ori  = add_calibration_adv2(lidar2cam ,cam2img, points_lidar)
            # miscalibrated_points2img_ori , mis_RT_ori , mis_KT_ori ,mis_K_ori = add_mis_calibration_ori(lidar2cam,cam2img, points_lidar,max_r=0.0,max_t=0.0)
            miscalibrated_points2img , extrinsic_perturb, lidar2img_original ,lidar2img_mis = add_mis_calibration_adv(lidar2img,lidar2cam,cam2img, points_lidar, max_r=1.0,max_t=0.010)

            point2img_gt.append(points2img) # lidar coordination 3d
            list_mis_RT.append(extrinsic_perturb) # lidar coordination 3d mis-calibration
            list_gt_KT.append(lidar2img)
            list_gt_KT_3by4.append(lidar2img_original)
            list_mis_KT.append(lidar2img_mis)
            
            ####### depth image display ######
            depth_gt, gt_uv,gt_z, valid_indices_gt= points2depthmap_gpu(points2img, results['img'][0].shape[0] ,results['img'][0].shape[1])
            # lidarOnImage_gt = torch.cat((gt_uv, gt_z.unsqueeze(1)), dim=1)
            # dense_depth_img_gt = dense_map_gpu_optimized(lidarOnImage_gt.T , results['img'][0].shape[1], results['img'][0].shape[0], 4)
            # dense_depth_img_gt = dense_depth_img_gt.to(dtype=torch.uint8)
            # dense_depth_img_color_gt = colormap(dense_depth_img_gt)

            depth_mis, uv,z,valid_indices = points2depthmap_gpu(miscalibrated_points2img, results['img'][0].shape[0] ,results['img'][0].shape[1])
            lidarOnImage_mis = torch.cat((uv, z.unsqueeze(1)), dim=1)
            dense_depth_img_mis = dense_map_gpu_optimized(lidarOnImage_mis.T , results['img'][0].shape[1], results['img'][0].shape[0], 4)
            dense_depth_img_mis = dense_depth_img_mis.to(dtype=torch.uint8)
            dense_depth_img_color_mis = colormap(dense_depth_img_mis)

            # lidar_depth_dense_gt.append(dense_depth_img_color_gt)
            lidar_depth_map_mis.append(dense_depth_img_color_mis)
            # gt_uvz.append(dense_depth_img_gt)
            lidar_depth_map_gt.append(depth_gt)
            # uvz.append(dense_depth_img_mis)
            
            # ###### input display ######
            # img = results['img'][cid]
            # # 이미지 데이터가 float 타입인 경우 0과 1 사이로 정규화
            # if img.dtype == np.float32 or img.dtype == np.float64:
            #     img = (img - img.min()) / (img.max() - img.min())
            # plt.figure(figsize=(20, 20))
            # plt.subplot(4,1,1)
            # plt.imshow(img)
            # plt.scatter(gt_uv[:, 0], gt_uv[:, 1], c=gt_z, s=0.5)
            # plt.title("input calibrated display", fontsize=10)

            # plt.subplot(4,1,2)
            # plt.imshow(img)
            # plt.scatter(uv[:, 0], uv[:, 1], c=z, s=0.5)
            # plt.title("input mis-calibrated display", fontsize=10)

            # disp_gt2 = dense_depth_img_color_gt.detach().cpu().numpy()
            # plt.subplot(4,1,3)
            # plt.imshow(disp_gt2, cmap='magma_r')
            # plt.title("gt display", fontsize=10)
            # plt.axis('off')

            # disp_mis2 = dense_depth_img_color_mis.detach().cpu().numpy()
            # plt.subplot(4,1,4)
            # plt.imshow(disp_mis2, cmap='magma')
            # plt.title("mis display", fontsize=10)
            # plt.axis('off')

            # # gt_gray = dense_depth_img_gt.detach().cpu().numpy()
            # # plt.subplot(3,2,5)
            # # plt.imshow(gt_gray, cmap='magma_r')
            # # plt.title("gt gray display", fontsize=10)
            # # plt.axis('off')

            # # mis_gray = dense_depth_img_mis.detach().cpu().numpy()
            # # plt.subplot(3,2,6)
            # # plt.imshow(mis_gray, cmap='magma_r')
            # # plt.title("mis gray display", fontsize=10)
            # # plt.axis('off')
            
            # # 전체 그림 저장
            # plt.tight_layout()
            # plt.savefig('load_pipeline.jpg', dpi=300, bbox_inches='tight')
            # plt.close()
            # print ("end of print")
        
        gt_KT = torch.stack(list_gt_KT)
        gt_KT_3by4 = torch.stack(list_gt_KT_3by4)
        mis_RT = torch.stack(list_mis_RT)
        mis_KT = torch.stack(list_mis_KT)
        lidar_depth_mis = torch.stack(lidar_depth_map_mis)
        lidar_depth_gt = torch.stack(lidar_depth_map_gt)
        lidar_depth_mis = lidar_depth_mis.permute(0, 3, 1, 2)
        # lidar_depth_gt = F.interpolate(lidar_depth_gt, size=[192, 640], mode="bilinear") # lidar 2d depth map input [192,640,1]
        # lidar_depth_mis = F.interpolate(lidar_depth_mis, size=[192, 640], mode="bilinear") 

        results['lidar_depth_gt']  = lidar_depth_gt
        results['lidar_depth_mis'] = lidar_depth_mis
        results['mis_KT'] = mis_KT
        results['mis_Rt'] = mis_RT
        results['gt_KT'] = gt_KT
        results['gt_KT_3by4'] = gt_KT_3by4

        return results
    
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



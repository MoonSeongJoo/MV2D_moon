import torch
import numpy as np
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.cm as cm
import scipy
import skimage
# from pypardiso import spsolve
from PIL import Image , ImageDraw
from COTR.utils import utils
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

def display_depth_maps(imgs, mis_calibrated_depth_map, sbs_img):
    # """
    # 이미지 위에 depth map과 mis_calibrated depth map을 오버레이하여 디스플레이하는 함수
    # """

    for cid in range(imgs.shape[0]):

        img_np = imgs.squeeze()[cid].cpu().detach().numpy()
        # depth_gt_np = depth_map.squeeze()[cid].cpu().detach().numpy() 
        depth_mis_np = mis_calibrated_depth_map.squeeze()[cid].cpu().detach().numpy()
        sbs_img_np = sbs_img.squeeze()[cid].cpu().detach().numpy()
        img_np = np.transpose(img_np,(1,2,0))
        # depth_gt_np = np.transpose(depth_gt_np,(1,2,0))
        depth_mis_np = np.transpose(depth_mis_np,(1,2,0))
        sbs_img_np = np.transpose(sbs_img_np,(1,2,0))
        # 이미지 데이터가 float 타입인 경우 0과 1 사이로 정규화
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            # depth_gt_np = (depth_gt_np - depth_gt_np.min()) / (depth_gt_np.max() - depth_gt_np.min())
            depth_mis_np = (depth_mis_np - depth_mis_np.min()) / (depth_mis_np.max() - depth_mis_np.min())
            sbs_img_np = (sbs_img_np - sbs_img_np.min()) / (sbs_img_np.max() - sbs_img_np.min())

        # input display
        ####### display input signal #########        
        plt.figure(figsize=(20, 20))
        plt.subplot(311)
        plt.imshow(img_np)
        plt.title("camera_input", fontsize=15)
        plt.axis('off')

        # plt.subplot(312)
        # plt.imshow( depth_gt_np, cmap='magma')
        # plt.title("calibrated_lidar_input", fontsize=15)
        # plt.axis('off') 

        plt.subplot(312)
        plt.imshow( depth_mis_np, cmap='magma')
        plt.title("mis-calibrated_lidar_input", fontsize=15)
        plt.axis('off')

        plt.subplot(313)
        plt.imshow( sbs_img_np)
        plt.title("sbs_img_input", fontsize=15)
        plt.axis('off')
        
        plt.tight_layout(pad=0)  # 여백 제거
        plt.savefig(f'raw_input_image_{cid+1}.png', dpi=300, bbox_inches='tight')
        plt.close('all')
    
    print ('display end')
    # ############ end of display input signal ###################

def normalize_uvz_points(points_lidar2img):
    
    normal_points_lidar2img = points_lidar2img.clone()
    normal_points_lidar2img[:, 0] = normal_points_lidar2img[:, 0]/1280
    normal_points_lidar2img[:, 1] = normal_points_lidar2img[:, 1]/192
    if normal_points_lidar2img[:, 2].numel() > 0:
        normal_points_lidar2img[:, 2] = (normal_points_lidar2img[:, 2]-torch.min(normal_points_lidar2img[:, 2]))\
            /(torch.max(normal_points_lidar2img[:, 2]) - torch.min(normal_points_lidar2img[:, 2]))
    else :
        normal_points_lidar2img[:, 2] = (normal_points_lidar2img[:, 2]-0)/(80 - 0)

    return normal_points_lidar2img

def scale_uvz_points(uvz_tensor, original_size=(512, 1408), target_size=(192, 640)):
    """
    UVZ 좌표를 이미지 스케일링 비율에 맞게 변환
    Args:
        uvz_tensor: (N, 3) 형태의 텐서 [u, v, z]
        original_size: (H, W) 원본 이미지 크기
        target_size: (H, W) 타겟 이미지 크기
    Returns:
        스케일링된 (N, 3) UVZ 텐서
    """
    # 스케일링 계수 계산 (높이, 너비)
    scale_h = target_size[0] / original_size[0]  # 192/512 = 0.375
    scale_w = target_size[1] / original_size[1]  # 640/1408 ≈ 0.4545
    
    # UV 좌표 스케일링 (깊이 z는 변경 없음)
    scaled_uvz = uvz_tensor.clone()
    scaled_uvz[:, 0] *= scale_w  # u 좌표(너비 방향) 스케일
    scaled_uvz[:, 1] *= scale_h  # v 좌표(높이 방향) 스케일
    
    return scaled_uvz

def inverse_scale_uvz_points(scaled_uvz, original_size=(512, 1408), target_size=(192, 640)):
    """
    스케일링된 UVZ 좌표를 원본 크기로 역변환
    Args:
        scaled_uvz: (N, 3) 형태의 텐서 [scaled_u, scaled_v, z]
        original_size: (H, W) 원본 이미지 크기 (스케일링 전)
        target_size: (H, W) 타겟 이미지 크기 (스케일링 후)
    Returns:
        역변환된 (N, 3) UVZ 텐서
    """
    # 역스케일링 계수 계산 (높이, 너비)
    inv_scale_h = original_size[0] / target_size[0]  # 512/192 ≈ 2.6667
    inv_scale_w = original_size[1] / target_size[1]  # 1408/640 = 2.2
    
    # 좌표 복원
    original_uvz = scaled_uvz.clone()
    original_uvz[:, 0] *= inv_scale_w  # u 좌표 복원
    original_uvz[:, 1] *= inv_scale_h  # v 좌표 복원
    
    return original_uvz

def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z

def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    # pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    # cam_intrinsic = cam_calib.numpy()
    # pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    pc_rotated = pc_rotated[:3, :].detach()
    cam_intrinsic = torch.tensor(cam_calib, dtype=torch.float32).cuda()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.t(), cam_intrinsic)
    
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0] ) & (pcl_z > 0)
    # mask1 = (pcl_uv[:, 1] < 188)

    pcl_uv_no_mask = pcl_uv
    # pcl_z_no_mask = pcl_z
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = torch.tensor(pcl_uv, dtype=torch.int64)
    # pcl_uv = pcl_uv.astype(np.uint32)
    # pcl_uv_no_mask  = pcl_uv_no_mask.astype(np.uint32) 
    
    pcl_z = pcl_z.reshape(-1, 1)

    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).cuda()
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z  
    depth_img = depth_img.permute(2, 0, 1)
    points_index = torch.arange(pcl_uv_no_mask.shape[0], device=pcl_uv_no_mask.device)[mask]

    # points_index = np.arange(pcl_uv_no_mask.shape[0])[mask]
    # points_index1 = np.arange(pcl_uv_no_mask.shape[0])[mask1]

    return depth_img, pcl_uv , pcl_z , points_index 

def lidar_project_depth_nuscenes(nusc,lidar_points, cam_data, real_shape):
    
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pc_points = lidar_points.cpu().numpy()
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc_points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc_points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    # points = view_points(lidar_points.points, np.array(cs_record['camera_intrinsic']), normalize=True)

    # 투영된 점들이 이미지 안에 있는지 확인하고, 그렇다면 그 위치에 따라 점들을 그립니다.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < real_shape[1]) # 이미지의 width 안에 있는지 확인
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < real_shape[0]) # 이미지의 height 안에 있는지 확인

    pcl_uv = points.transpose(1,0)[:,:2]
    pcl_z = points.transpose(1,0)[:,2]
    pcl_uv_no_mask = pcl_uv
    pcl_uv = pcl_uv[mask]    
    # pcl_z = pcl_z[mask]
    pcl_z = depths[mask]
    pcl_uv = torch.tensor(pcl_uv, dtype=torch.int32).cuda()
    pcl_z = torch.tensor(pcl_z, dtype=torch.float32).cuda()
    pcl_z = pcl_z.reshape(-1, 1)
    
    depth_img = np.zeros((real_shape[0], real_shape[1], 1))
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).cuda()
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z  
    depth_img = depth_img.permute(2, 0, 1)
    points_index = torch.arange(pcl_uv_no_mask.shape[0], device='cuda')[mask]

    return depth_img, pcl_uv , pcl_z , points_index

def transform_gt(nusc, lidar_data, lidar_points ,cam_data) :
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    lidar_cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_points.rotate(Quaternion(lidar_cs_record['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_cs_record['translation']))

    # Second step: transform from ego to the global frame.
    lidar_poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_points.rotate(Quaternion(lidar_poserecord['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
    lidar_points.translate(-np.array(poserecord['translation']))
    lidar_points.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_points.translate(-np.array(cs_record['translation']))
    lidar_points.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    return lidar_points
    
# def trim_corrs(in_corrs, num_kp=100):
#     length = in_corrs.shape[0]
# #         print ("number of keypoint before trim : {}".format(length))
#     if length >= num_kp:
#         mask = np.random.choice(length, num_kp)
#         return in_corrs[mask]
#     elif length==0:
#         return np.random.rand(num_kp, 3)
#     else:
#         mask = np.random.choice(length, num_kp - length)
#         return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

def trim_corrs(in_corrs, num_kp=100):
    device = in_corrs.device  # 원본 텐서 디바이스 유지
    length = in_corrs.shape[0]
    
    if length >= num_kp:
        # 무작위 선택 (비복원 추출)
        mask = torch.randperm(length, device=device)[:num_kp]
        return in_corrs[mask]
    # elif length == 0:
    #     # 0~1 범위 랜덤 텐서 생성 [num_kp, 6]
    #     return torch.rand(num_kp, 6, device=device)
    else:
        # 부족분 채우기 (복원 추출)
        mask = torch.randint(0, length, (num_kp - length,), device=device)
        return torch.cat([in_corrs, in_corrs[mask]], dim=0)

# def process_queries(corrs, sbs_img, num_points=100):
#     device = corrs.device
    
#     # 1. 고유 카메라 인덱스 추출 및 정렬
#     unique_cams, counts = torch.unique(corrs[:, 0], 
#                                      sorted=False, 
#                                      return_counts=True)
    
#     # 2. 이미지 선택
#     selected_imgs = sbs_img[unique_cams.long()]
    
#     # 3. 카메라별 쿼리 처리
#     grouped_queries = []
#     camera_indices = []
    
#     for cam_idx, cam in enumerate(unique_cams):
#         # 현재 카메라 쿼리 추출
#         mask = (corrs[:, 0] == cam)
#         cam_queries = corrs[mask, 1:]  # [N, 4] (obj_id, u, v, z)
        
#         # 트리밍/패딩
#         if cam_queries.size(0) >= num_points:
#             idx = torch.randperm(cam_queries.size(0))[:num_points]
#             selected = cam_queries[idx]
#         else:
#             idx = torch.randint(0, cam_queries.size(0), (num_points - cam_queries.size(0),))
#             selected = torch.cat([cam_queries, cam_queries[idx]], dim=0)
        
#         # 카메라 인덱스 태깅 및 저장
#         tagged_queries = torch.cat([
#             torch.full((num_points, 1), cam_idx, device=device, dtype=torch.float),
#             selected
#         ], dim=1)  # [num_points, 5] (local_cam_idx,ob_id, u, v, z)
        
#         grouped_queries.append(tagged_queries)
#         camera_indices.append(cam)  # 원본 카메라 인덱스 저장

#     # 4. 배치 차원 생성 및 원본 카메라 인덱스 반환
#     processed_queries = torch.stack(grouped_queries, dim=0)  # [B_sel, 100, 4]
#     original_camera_ids = torch.stack(camera_indices)        # [B_sel]
    
#     return selected_imgs, processed_queries, original_camera_ids

def process_queries(corrs, sbs_img, num_points=100):
    device = corrs.device
    
    # 입력 데이터가 비어 있는 경우 빈 텐서 반환
    if corrs.numel() == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),  # selected_imgs
            torch.empty(0, num_points, 5, device=device),       # processed_queries
            torch.empty(0, device=device)                       # original_camera_ids
        )
    
    # 1. 고유 카메라 인덱스 추출
    unique_cams, counts = torch.unique(corrs[:, 0], 
                                     sorted=False, 
                                     return_counts=True)
    
    # unique_cams가 비어 있는 경우 처리
    if len(unique_cams) == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 5, device=device),
            torch.empty(0, device=device)
        )
    
    # 2. 이미지 선택
    selected_imgs = sbs_img[unique_cams.long()]
    
    # 3. 카메라별 쿼리 처리
    grouped_queries = []
    camera_indices = []
    
    for cam_idx, cam in enumerate(unique_cams):
        mask = (corrs[:, 0] == cam)
        cam_queries = corrs[mask, 1:]  # [N, 4]
        
        # 데이터 트리밍/패딩
        if cam_queries.size(0) >= num_points:
            idx = torch.randperm(cam_queries.size(0))[:num_points]
            selected = cam_queries[idx]
        else:
            padding_size = num_points - cam_queries.size(0)
            selected = torch.cat([
                cam_queries,
                cam_queries[torch.randint(0, cam_queries.size(0), (padding_size,))]
            ], dim=0)
        
        # 카메라 인덱스 추가
        tagged_queries = torch.cat([
            torch.full((num_points, 1), cam_idx, device=device, dtype=corrs.dtype),
            selected
        ], dim=1)
        
        grouped_queries.append(tagged_queries)
        camera_indices.append(cam)
    
    # 4. 배치 차원 생성
    processed_queries = torch.stack(grouped_queries, dim=0)
    original_camera_ids = torch.stack(camera_indices)
    
    return selected_imgs, processed_queries, original_camera_ids

def process_queries_adv(corrs, sbs_img, num_points=100):
    device = corrs.device
    
    # 입력 데이터가 비어 있는 경우 빈 텐서 반환
    if corrs.numel() == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 5, device=device),
            torch.empty(0, device=device)
        )
    
    # 1. 고유 카메라 인덱스 추출
    unique_cams, counts = torch.unique(corrs[:, 0], sorted=False, return_counts=True)
    
    # unique_cams가 비어 있는 경우 처리
    if len(unique_cams) == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 5, device=device),
            torch.empty(0, device=device)
        )
    
    # 2. 이미지 선택
    selected_imgs = sbs_img[unique_cams.long()]
    
    # 3. 카메라별 쿼리 처리
    grouped_queries = []
    camera_indices = []
    
    for cam_idx, cam in enumerate(unique_cams):
        mask = (corrs[:, 0] == cam)
        cam_queries = corrs[mask, 1:]  # [N, 4]
        
        # 객체별 샘플링 로직 (핵심 수정 부분)
        if cam_queries.size(0) >= num_points:
            # 객체 ID 추출 및 분포 계산
            object_ids = cam_queries[:, 0]
            unique_objs, obj_counts = torch.unique(object_ids, return_counts=True)
            
            # 객체 빈도순 정렬 (빈도 높은 객체 우선)
            sorted_objs = unique_objs[torch.argsort(-obj_counts)]
            
            # 객체별 최소 1개 샘플링 보장
            selected_indices = []
            for obj in sorted_objs:
                obj_mask = (object_ids == obj)
                obj_indices = torch.where(obj_mask)[0]
                if len(selected_indices) < num_points:
                    idx = torch.randint(0, len(obj_indices), (1,))
                    selected_indices.append(obj_indices[idx])
            
            # 남은 슬롯 채우기
            remaining = num_points - len(selected_indices)
            if remaining > 0:
                all_indices = torch.arange(len(object_ids), device=device)
                mask = torch.ones(len(object_ids), dtype=torch.bool, device=device)
                
                # Concatenate selected indices into a single tensor
                if selected_indices:
                    selected_indices = torch.cat(selected_indices).to(device)
                else:
                    selected_indices = torch.tensor([], dtype=torch.long, device=device)
                
                mask[selected_indices] = False

                remaining_indices = all_indices[mask]
                if len(remaining_indices) > 0:
                    idx = torch.randperm(len(remaining_indices))[:remaining]
                    selected_remaining = remaining_indices[idx]
                    selected_indices = torch.cat([selected_indices, selected_remaining])
            
            # 최종 선택 및 충돌 방지
            selected_indices = torch.tensor(selected_indices[:num_points], device=device)
            selected = cam_queries[selected_indices]
        else:
            # 기존 패딩 로직 (객체 다양성 고려)
            padding_size = num_points - cam_queries.size(0)
            selected = torch.cat([
                cam_queries,
                cam_queries[torch.randint(0, cam_queries.size(0), (padding_size,))]
            ], dim=0)
        
        # 카메라 인덱스 추가
        tagged_queries = torch.cat([
            torch.full((num_points, 1), cam_idx, device=device, dtype=corrs.dtype),
            selected
        ], dim=1)
        
        grouped_queries.append(tagged_queries)
        camera_indices.append(cam)
    
    # 4. 배치 차원 생성
    processed_queries = torch.stack(grouped_queries, dim=0)
    original_camera_ids = torch.stack(camera_indices)
    
    return selected_imgs, processed_queries, original_camera_ids

# def trim_corrs_torch(in_corrs, num_kp=100):
#     length = in_corrs.shape[0]
    
#     if length == 0 :
#         reduced_corrs = torch.rand((num_kp, in_corrs.shape[1]), dtype=in_corrs.dtype, device=in_corrs.device) * 1e-6
#     elif in_corrs is None :
#         reduced_corrs =torch.rand((num_kp, in_corrs.shape[1]), dtype=in_corrs.dtype, device=in_corrs.device) * 1e-6
#     elif length >= num_kp:
#         mask = torch.randperm(length)[:num_kp]
#         reduced_corrs =in_corrs[mask]
#     else:
#         mask = torch.randperm(length).repeat(num_kp // length + 1)[:num_kp]
#         reduced_corrs = in_corrs[mask]
    
#     return reduced_corrs 

def trim_corrs_torch(in_corrs, num_kp=100):
    if in_corrs is None or in_corrs.shape[0] == 0:
        # Handle None or empty input
        return torch.rand((num_kp, 6), dtype=torch.float32, device='cuda') * 1e-6

    length = in_corrs.shape[0]
    device = in_corrs.device

    if length >= num_kp:
        mask = torch.randperm(length, device=device)[:num_kp]
    else:
        mask = torch.randperm(length, device=device).repeat(num_kp // length + 1)[:num_kp]

    return in_corrs[mask]


def resize_points(query_xyz):
    original_height, original_width = 900,1600
    target_height, target_width = 192,640

    # 스케일 계산
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    # 점 좌표 조정
    resized_points = query_xyz.clone()
    resized_points[..., 0] *= scale_w  # x 좌표 조정
    resized_points[..., 1] *= scale_h  # y 좌표 조정

    return resized_points

def farthest_point_sampling(points, k):
    """
    Args:
        points (torch.Tensor): (N, 3) shape의 포인트 집합
        k (int): 선택할 중심 포인트의 개수
    Returns:
        torch.Tensor: (k, 3) shape의 선택된 중심 포인트 좌표
        torch.Tensor: (k) shape의 선택된 중심 포인트 인덱스
    """
    N, _ = points.shape
    centroids = torch.zeros(k, dtype=torch.long, device=points.device)
    distance = torch.ones(N, device=points.device) * 1e10

    # 첫 번째 중심 포인트를 무작위로 선택
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=points.device)

    for i in range(k):
        # 가장 먼 지점을 중심 포인트로 선택
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)

        # 선택한 중심 포인트와 다른 모든 포인트 간의 거리 계산
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]

        # 가장 먼 포인트를 찾는다
        farthest = torch.max(distance, dim=0)[1]

    # 선택된 중심 포인트 좌표 및 인덱스 반환
    return centroids ,points[centroids]

def knn(x, y ,k):
# #         print (" x shape = " , x.shape)
#         inner = -2*torch.matmul(x.transpose(-2, 1), x)
#         xx = torch.sum(x**2, dim=1, keepdim=True)
# #         print (" xx shape = " , x.shape)
#         pairwise_distance = -xx - inner - xx.transpose(4, 1)
    # mask_x = (x[: , 2] > 0.5) & (x[: , 2] < 0.8)
    # mask_y = (y[: , 2] > 0.5) & (y[: , 2] < 0.8)
    # x1 = x[mask_x]
    # y1 = y[mask_y]
    # mask_x1= np.in1d(mask_x,mask_y)
    # mask_y1= np.in1d(mask_y,mask_x)
    # x2 = x[mask_x1]
    # y2 = y[mask_y1]
    # x2 = torch.from_numpy(x2)  # NumPy 배열을 PyTorch Tensor로 변환
    # y2 = torch.from_numpy(y2)  # NumPy 배열을 PyTorch Tensor로 변환
    # pairwise_distance = F.pairwise_distance(x,y)
    
    # #### monitoring x/y point #####################
    # print ("x2 x_point min =" , torch.min(x[:,0]))
    # print ("x2 x_point max =" , torch.max(x[:,0]))
    # print ("y2 x_point min =" , torch.min(y[:,0]))
    # print ("y2 x_point max =" , torch.max(y[:,0]))
    # print ("x2 y_point min =" , torch.min(x[:,1]))
    # print ("x2 y_point max =" , torch.max(x[:,1]))
    # print ("y2 y_point min =" , torch.min(y[:,1]))
    # print ("y2 y_point max =" , torch.max(y[:,1]))
    # print ("x2 depth min =" , torch.min(x[:,2]))
    # print ("x2 depth max =" , torch.max(x[:,2]))
    # print ("y2 depth min =" , torch.min(y[:,2]))
    # print ("y2 depth max =" , torch.max(y[:,2]))
    # ##############################################
    
    # 일정 depth range (min_depth, max_depth)
    min_depth = 0.05
    max_depth = 0.2
    
    # y[:, 2] = 1 - y[:, 2] # 세 번째 열 값 반전
    # min_depth <= depth <= max_depth 인 point들의 인덱스를 구합니다.
    depth_mask1 = (x[:, 2] >= min_depth) & (x[:, 2] <= max_depth) # & (x[:,1] >= 0.6 )
    depth_mask2 = (y[:, 2] >= min_depth) & (y[:, 2] <= max_depth) # & (y[:,1] >= 0.6 )
    # depth_indices1 = np.where(depth_mask1)[0]
    # depth_indices2 = np.where(depth_mask2)[0]
    depth_indices1 = torch.nonzero(depth_mask1).squeeze()
    depth_indices2 = torch.nonzero(depth_mask2).squeeze()

    x1 = x[depth_indices1]
    y1 = y[depth_indices2]

    # mask_x1= np.in1d(depth_indices1,depth_indices2)
    # mask_y1= np.in1d(depth_indices2,depth_indices1)
    mask_x1 = (depth_indices1.view(-1, 1)== depth_indices2.view(1, -1)).any(dim=1)
    mask_y1 = (depth_indices2.view(-1, 1) == depth_indices1.view(1, -1)).any(dim=1)
    # mask_x1 = torch.tensor([elem in depth_indices2.cpu().numpy() for elem in depth_indices1.cpu().numpy()], device=x.device, dtype=torch.bool)
    # mask_y1 = torch.tensor([elem in depth_indices1.cpu().numpy() for elem in depth_indices2.cpu().numpy()], device=y.device, dtype=torch.bool)

    x2 = x1.index_select(0, torch.nonzero(mask_x1).squeeze())
    y2 = y1.index_select(0, torch.nonzero(mask_y1).squeeze())
    # x2 = x1[mask_x1]
    # y2 = y1[mask_y1]
    
    if x2.shape[0] <= k :
        # x2 = torch.zeros(k, 3 , device=x.device)
        # y2 = torch.zeros(k, 3,  device=y.device)
        ### 부족하면 무조건 랜덤 수 채우기
        x2 = torch.rand(k, 3).cuda()
        y2 = torch.rand(k, 3).cuda()
            
  
    #### 유사한 포인트 뽑기 using KNN #####
    pairwise_distance = F.pairwise_distance(x2, y2)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    # top_indices = torch.topk(pairwise_distance.flatten(), k=k, largest=False)
    # top_indices = top_indices.indices
    # indices = np.unravel_index(top_indices, pairwise_distance.shape)
    # top_indices = np.asarray(top_indices).T
    
    #### 가장 먼 포인트 들 뽑기 #########
    # idx ,_ = farthest_point_sampling(x2,k)

    ########## 랜덤으로 포인트 뽑기 #########
    # idx = torch.randperm(x2.shape[0])[:k]

    top_x = x2[idx]
    top_y = y2[idx]
    # print ("x point of z =" , top_x[3])
    # print ("y point of z =" , top_y[3])
    # top_y[:, 2] =  1- top_y[:, 2] # 세 번째 열의 값에서 1을 빼기 
    # print ("y point of rev z =" , top_y[3])
    
    corrs = torch.cat([top_x,top_y] ,dim=1) 
        
    return idx , corrs

def two_images_side_by_side_np(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
    h, w, c = img_a.shape
#         b,h, w, c = img_a.shape
    canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
    canvas[:, 0 * w:1 * w, :] = img_a
    canvas[:, 1 * w:2 * w, :] = img_b
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
#         canvas[:, :, 0 * w:1 * w, :] = img_a.cpu().numpy()
#         canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()

    #canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
    #canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
    return canvas

def two_images_side_by_side(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype

    img_a = img_a.permute(0,2,3,1)
    img_b = img_b.permute(0,2,3,1)
    b, h, w, c = img_a.shape
#         canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
#         canvas[:, 0 * w:1 * w, :] = img_a
#         canvas[:, 1 * w:2 * w, :] = img_b
    canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
    canvas[:, :, 0 * w:1 * w, :] = img_a.detach().cpu().numpy()
    # canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()
    canvas[:, :, 1 * w:2 * w, :] = img_b.detach().cpu().numpy()

#         canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
#         canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
    return canvas

# From Github https://github.com/balcilar/DenseDepthMap
def dense_map(Pts ,n, m, grid):
    ng = 2 * grid + 1

    # mX = np.zeros((m,n)) + np.float("inf")
    # mY = np.zeros((m,n)) + np.float("inf")
    # mD = np.zeros((m,n))

    # mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    # mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    # mD[np.int32(Pts[1]),np.int32(Pts[1])] = Pts[2]

    # KmX = np.zeros((ng, ng, m - ng, n - ng))
    # KmY = np.zeros((ng, ng, m - ng, n - ng))
    # KmD = np.zeros((ng, ng, m - ng, n - ng))

    mX = torch.full((m, n), float('inf'), dtype=torch.float32, device='cuda')
    mY = torch.full((m, n), float('inf'), dtype=torch.float32, device='cuda')
    mD = torch.zeros((m, n), dtype=torch.float32, device='cuda')

    mX_idx = torch.tensor(Pts[1], dtype=torch.int64)
    mY_idx = torch.tensor(Pts[0], dtype=torch.int64)

    mX[mX_idx, mY_idx] = Pts[0] - torch.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - torch.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')

    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    # S = np.zeros_like(KmD[0,0])
    # Y = np.zeros_like(KmD[0,0])
    S = torch.zeros_like(KmD[0, 0])
    Y = torch.zeros_like(KmD[0, 0])

    for i in range(ng):
        for j in range(ng):
            # s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            s = 1 / torch.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i,j]
            S = S + s

    S[S == 0] = 1
    # out = np.zeros((m,n))
    out = torch.zeros((m, n), dtype=torch.float32, device='cuda')
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out 

def colormap(disp):
    """"Color mapping for disp -- [H, W] -> [3, H, W]"""
    disp_np = disp.cpu().numpy()        # tensor -> numpy
    # disp_np = disp
    # vmax = np.percentile(disp_np, 95)
    vmin = disp_np.min()
    vmax = disp_np.max()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    # return colormapped_im.transpose(2, 0, 1)
    colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32).cuda()
    # colormapped_tensor = torch.from_numpy(colormapped_im).
    return colormapped_tensor

# corr dataset generation 
def corr_gen( gt_points_index, points_index, gt_uv, uv , num_kp = 500) :
    
    inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
    inter_uv_mask    = np.in1d(points_index , gt_points_index)
    gt_uv = gt_uv[inter_gt_uv_mask]
    uv    = uv[inter_uv_mask] 
    corrs = np.concatenate([gt_uv, uv], axis=1)
    corrs = torch.tensor(corrs)
    
    ## corrs 384*1280 image(original image shape) normalization
    corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:, 2] = (0.5*corrs[:, 2])/1280 + 0.5        
    corrs[:, 3] = (0.5*corrs[:, 3])/384   
    
    if corrs.shape[0] <= num_kp :
        corrs = torch.zeros(num_kp, 4)
        corrs[:, 2] = corrs[:, 2] + 0.5

    corrs_knn_idx = knn(corrs[:,:2], corrs[:,2:], num_kp) # knn 2d point-cloud trim
    corrs = corrs[corrs_knn_idx]               

    assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
    assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
    assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
    assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
    
    return corrs

def corr_gen_withZ( gt_points_index, points_index, gt_uv, uv , gt_z, z, origin_img_shape, resized_shape, num_kp = 500) :
    
    #only numpy operation
    # inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
    # inter_uv_mask    = np.in1d(points_index , gt_points_index)

    inter_gt_uv_mask = torch.tensor(np.in1d(gt_points_index.cpu().numpy(), points_index.cpu().numpy()), device='cuda')
    inter_uv_mask = torch.tensor(np.in1d(points_index.cpu().numpy(), gt_points_index.cpu().numpy()), device='cuda')
    gt_uv = gt_uv[inter_gt_uv_mask]
    uv    = uv[inter_uv_mask] 
    gt_z = gt_z[inter_gt_uv_mask]
    z    = z[inter_uv_mask] 
    # gt_uvz = np.concatenate([gt_uv,gt_z], axis=1)
    # uvz= np.concatenate([uv,z],axis=1)
    # corrs = np.concatenate([gt_uvz, uvz], axis=1)
    # corrs = torch.tensor(corrs)
    gt_uvz = torch.cat([gt_uv, gt_z], dim=1)
    uvz = torch.cat([uv, z], dim=1)
    corrs = torch.cat([gt_uvz, uvz], dim=1)

    # gt_points = torch.tensor(gt_uvz)
    # target_points = torch.tensor(uvz)
    # scale_img = np.array (resized_shape) / np.array(origin_img_shape) 
    
    # #### monitoring x/y point #####################
    # print ("origin gt x_point min =" ,     torch.min(corrs[:,0]))
    # print ("origin gt x_point max =" ,     torch.max(corrs[:,0]))
    # print ("origin target x_point min =" , torch.min(corrs[:,3]))
    # print ("origin target x_point max =" , torch.max(corrs[:,3]))
    # print ("origin gt y_point min =" ,     torch.min(corrs[:,1]))
    # print ("origin gt y_point max =" ,     torch.max(corrs[:,1]))
    # print ("origin target y_point min =" , torch.min(corrs[:,1]))
    # print ("origin target y_point max =" , torch.max(corrs[:,1]))
    # print ("origin gt depth min =" ,       torch.min(corrs[:,2]))
    # print ("origin gt depth max =" ,       torch.max(corrs[:,2]))
    # print ("origin target depth min =" ,   torch.min(corrs[:,2]))
    # print ("origin target depth max =" ,   torch.max(corrs[:,2]))
    # ##############################################
    
    # corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:, 0] = corrs[:, 0]/origin_img_shape[1] 
    # corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:, 1] = corrs[:, 1]/origin_img_shape[0] 
    if corrs[:, 2].numel() > 0:
        corrs[:, 2] = (corrs[:, 2]-torch.min(corrs[:, 2]))/(torch.max(corrs[:, 2]) - torch.min(corrs[:, 2]))
    else :
        corrs[:, 2] = (corrs[:, 2]-0)/(80 - 0)
    # corrs[:, 3] = (0.5*corrs[:, 3])/1280 + 0.5
    corrs[:, 3] = corrs[:, 3]/origin_img_shape[1]         
    # corrs[:, 4] = (0.5*corrs[:, 4])/384
    corrs[:, 4] = corrs[:, 4]/origin_img_shape[0]
    if corrs[:, 5].numel() > 0:
        corrs[:, 5] = (corrs[:, 5]-torch.min(corrs[:, 5]))/(torch.max(corrs[:, 5]) - torch.min(corrs[:, 5])) 
    else :
        corrs[:, 5] = (corrs[:, 5]-0)/(80 - 0)

    # #### monitoring x/y point #####################
    # print ("normalized gt x_point min =" ,     torch.min(corrs[:,0]))
    # print ("normalized gt x_point max =" ,     torch.max(corrs[:,0]))
    # print ("normalized target x_point min =" , torch.min(corrs[:,3]))
    # print ("normalized target x_point max =" , torch.max(corrs[:,3]))
    # print ("normalized gt y_point min =" ,     torch.min(corrs[:,1]))
    # print ("normalized gt y_point max =" ,     torch.max(corrs[:,1]))
    # print ("normalized target y_point min =" , torch.min(corrs[:,1]))
    # print ("normalized target y_point max =" , torch.max(corrs[:,1]))
    # print ("normalized gt depth min =" ,       torch.min(corrs[:,2]))
    # print ("normalized gt depth max =" ,       torch.max(corrs[:,2]))
    # print ("normalized target depth min =" ,   torch.min(corrs[:,2]))
    # print ("normalized target depth max =" ,   torch.max(corrs[:,2]))
    # ##############################################

    if corrs.shape[0] <= num_kp :
        # corrs = torch.zeros(num_kp, 6)
        diff = num_kp - corrs.shape[0]
        rand_values = torch.randn(diff, 6).cuda()
        corrs = torch.cat([corrs, rand_values], dim=0)
        # target_points = torch.zeros(num_kp, 3)
        # corrs[:, 2] = corrs[:, 2] + 0.5 # for only uv matching
        # corrs[:, 3] = corrs[:, 3] + 0.5 # for uvz matching

    corrs_knn_idx ,corrs_prev = knn(corrs[:,:3], corrs[:,3:], num_kp) # knn 2d point-cloud trim

    corrs = corrs[corrs_knn_idx]   
    corrs1 = corrs_prev
    # corrs = corrs[z_mask]    
    # corrs = torch.cat([top_gt_points,top_target_points],dim=1)

    # assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
    # assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
    # assert (0.0 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
    # assert (0.5 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
    # assert (0.0 <= corrs[:, 4]).all() and (corrs[:, 4] <= 1.0).all()
    # assert (0.0 <= corrs[:, 5]).all() and (corrs[:, 5] <= 1.0).all()
    
    return corrs1

def random_mask(sbs_img,grid_size=(32, 32), mask_value=0):
    # sbs_img shape: [batch, channel, height, width]
    batch_size, _, height, width = sbs_img.shape
    mask = torch.ones_like(sbs_img)

    grid_height, grid_width = grid_size

    for i in range(height // grid_height):
        for j in range(width // grid_width):
            if torch.rand(1) > 0.75:  # Randomly choose whether to mask this grid or not
                # Apply the mask to the corresponding area in the image
                mask[:, :, i*grid_height:(i+1)*grid_height,j*grid_width:(j+1)*grid_width] = mask_value

    return sbs_img * mask


def draw_points(img, query , mode='640*192'):

    img = Image.fromarray(np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 포인트 그리기
    if mode=='640*192':
        query *= np.array([640,192])
    elif mode=='320_*92':
        query *= np.array([320,192])
    
    for (x, y) in query:
        draw.ellipse((x-1, y-1, x+1, y+1), fill='red', outline='red')
    
    return np.array(img)

def draw_points_torch(img, query, mode='640*192'):
    # 이미지 정규화 및 PIL 이미지로 변환
    img = img.clone()  # 텐서 복사
    img = (img - img.min()) / (img.max() - img.min()) * 255  # 정규화 후 0-255로 스케일링
    img = img.byte().cpu().numpy()  # 텐서를 NumPy로 변환
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    # 포인트 스케일링
    if mode == '640*192':
        query = query * torch.tensor([640, 192], device=query.device)
    elif mode == '320*192':
        query = query * torch.tensor([320, 192], device=query.device)

    # 포인트 그리기
    for (x, y) in query.cpu().numpy():
        draw.ellipse((x-1, y-1, x+1, y+1), fill='red', outline='red')

    return torch.tensor(np.array(img))  # 결과를 다시 텐서로 변환


def draw_center_point(self, img):
    
    img = Image.fromarray(np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 이미지 크기 구하기
    w, h = img.size

    # 정중앙 위치 계산하기
    query_x = w // 2
    query_y = h // 2
    radius = 1
    # 포인트 그리기
    draw.ellipse((query_x-radius, query_y-radius, query_x+radius, query_y+radius), fill='red', outline='red')

    return np.array(img)

def draw_corrs(imgs, corrs, col=(255, 0, 0)):
    imgs = utils.torch_img_to_np_img(imgs)
    out = []
    
    # 삭제하려는 열의 인덱스 리스트
    cols_to_remove = [2, 5] # Python은 0부터 시작하기 때문에 
    # 유지하려는 열들만 선택합니다.
    cols_to_keep = [i for i in range(corrs.shape[2]) if i not in cols_to_remove]
    # index_select 함수를 사용하여 해당 열들만 선택합니다.
    corrs_shrink = corrs[:,:,cols_to_keep]

    for img, corr in zip(imgs, corrs_shrink):
        img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
#             corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
        corr *= np.array([1280,192,1280,192])
        for c in corr:
            draw.line(c, fill=col)
        out.append(np.array(img))
    out = np.array(out) / 255.0
    return utils.np_img_to_torch_img(out) , out 

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output 

def find_depthmap_z(detections, depth_map):
    result_list = []
    device = depth_map.device  # depth_map의 디바이스를 기준으로 설정
    batch_size,num_cam,h,w=depth_map.shape
    depth_map_re = depth_map.view(batch_size*num_cam,h,w)

    for cid in range(num_cam):
        # # x 중심점 계산: (x_min + x_max) / 2
        center_x = (detections[cid][:, 0] + detections[cid][:, 2]) / 2
        # y 중심점 계산: (y_min + y_max) / 2
        center_y = (detections[cid][:, 1] + detections[cid][:, 3]) / 2
        # 중심점 좌표를 하나의 텐서로 결합
        center_points = torch.stack((center_x, center_y), dim=1)
        center_points_int = center_points.long()
        z_values = []

        for x, y in center_points_int:
            if 0 <= x < depth_map_re[cid].shape[1] and 0 <= y < depth_map_re[cid].shape[0]:
                z_values.append(depth_map_re[cid][y, x])  # depth map에서 값 추출
            else:
                z_values.append(torch.tensor(float('nan')).to(device))  # 범위를 벗어난 경우 NaN 할당
        
        z_values = torch.stack(z_values).unsqueeze(1) # z 값을 열 벡터로 변환
        concat_result = torch.cat((center_points_int, z_values), dim=1)  # x, y, z 결합
        result_list.append(concat_result)

    return result_list

def find_all_depthmap_z(detections, depth_map):
    result_list = []
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    for cid in range(num_cam):
        camera_results = []
        for bbox in detections[cid]:
            x_min, y_min, x_max, y_max = bbox[:4].long()
            
            # 바운딩 박스가 이미지 범위를 벗어나지 않도록 조정
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)
            
            # 바운딩 박스 내의 모든 포인트에 대해 z값 추출
            bbox_depth = depth_map_re[cid, y_min:y_max+1, x_min:x_max+1]
            
            # x, y 좌표 생성
            y_coords, x_coords = torch.meshgrid(torch.arange(y_min, y_max+1), torch.arange(x_min, x_max+1))
            coords = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).to(device)
            
            # z값과 좌표 결합
            z_values = bbox_depth.flatten().unsqueeze(1)
            bbox_points = torch.cat((coords, z_values), dim=1)
            
            camera_results.append(bbox_points)
        
        result_list.append(camera_results)

    return result_list

def find_all_depthmap_z_adv(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    all_bbox_points = []

    for bbox in detections:
        cid = int(bbox[0])  # Get camera ID
        x_min, y_min, x_max, y_max = bbox[1:5].long()
        
        # 바운딩 박스가 이미지 범위를 벗어나지 않도록 조정
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w - 1, x_max)
        y_max = min(h - 1, y_max)
        
        # 바운딩 박스 내의 모든 포인트에 대해 z값 추출
        bbox_depth = depth_map_re[cid, y_min:y_max+1, x_min:x_max+1]
        
        # x, y 좌표 생성
        y_coords, x_coords = torch.meshgrid(torch.arange(y_min, y_max+1), torch.arange(x_min, x_max+1), indexing='ij')
        coords = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).to(device)
        
        # z값과 좌표 결합
        z_values = bbox_depth.flatten().unsqueeze(1)
        bbox_points = torch.cat((coords, z_values), dim=1)
        
        all_bbox_points.append(bbox_points)

    if all_bbox_points:
        # 모든 포인트를 하나의 텐서로 결합
        combined_points = torch.cat(all_bbox_points, dim=0)
        
        # 중복된 (x, y) 좌표 제거 및 해당하는 z 값 유지
        unique_coords, unique_indices = torch.unique(combined_points[:, :2], dim=0, return_inverse=True)
        unique_z_values = torch.zeros(unique_coords.shape[0], 1, device=device)
        
        # 중복된 좌표의 z 값 평균 계산
        for i in range(unique_coords.shape[0]):
            unique_z_values[i] = combined_points[unique_indices == i, 2].mean()
        
        # 고유한 좌표와 해당하는 z 값을 결합
        return torch.cat((unique_coords, unique_z_values), dim=1)
    else:
        return torch.empty((0, 3), device=device)

def find_nonzero_depthmap_z(detections, depth_map):
    result_list = []
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    for cid in range(num_cam):
        camera_results = []
        for bbox in detections[cid]:
            x_min, y_min, x_max, y_max = bbox[:4].long()
            
            # 바운딩 박스가 이미지 범위를 벗어나지 않도록 조정
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)
            
            # 바운딩 박스 내의 모든 포인트에 대해 z값 추출
            bbox_depth = depth_map_re[cid, y_min:y_max+1, x_min:x_max+1]
            
            # x, y 좌표 생성
            y_coords, x_coords = torch.meshgrid(torch.arange(y_min, y_max+1 ,device=device), torch.arange(x_min, x_max+1,device=device))
            coords = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1)
            
            # z값이 0이 아닌 포인트만 선택
            nonzero_mask = bbox_depth.flatten() != 0
            nonzero_coords = coords[nonzero_mask]
            nonzero_z = bbox_depth.flatten()[nonzero_mask].unsqueeze(1)
            
            # z값과 좌표 결합
            bbox_points = torch.cat((nonzero_coords, nonzero_z), dim=1)
            
            camera_results.append(bbox_points)
        
        result_list.append(camera_results)

    return result_list

def find_rois_depthmap_z(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    cam_indices = detections[:, 0].long()
    bboxes = detections[:, 1:]
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w - 1)
    y_min = torch.clamp(y_min, 0, h - 1)
    x_max = torch.clamp(x_max, 0, w - 1)
    y_max = torch.clamp(y_max, 0, h - 1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i]
        
        center_x = (x_min[i] + x_max[i]) // 2
        center_y = (y_min[i] + y_max[i]) // 2
        
        center_z = depth_map_re[cid, center_y, center_x]
        
        if center_z != 0:
            result.append(torch.tensor([center_x, center_y, center_z], device=device))
        else:
            result.append(torch.tensor([center_x, center_y, float('nan')], device=device))

    return torch.stack(result)

def find_rois_nonzero_z(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    cam_indices = detections[:, 0].long()
    bboxes = detections[:, 1:]
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w - 1)
    y_min = torch.clamp(y_min, 0, h - 1)
    x_max = torch.clamp(x_max, 0, w - 1)
    y_max = torch.clamp(y_max, 0, h - 1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i]
        
        center_x = (x_min[i] + x_max[i]) // 2
        center_y = (y_min[i] + y_max[i]) // 2
        
        center_z = depth_map_re[cid, center_y, center_x]
        
        if center_z == 0:
            bbox_width = x_max[i] - x_min[i] + 1
            bbox_height = y_max[i] - y_min[i] + 1
            
            x_start = max(0, center_x - bbox_width // 2)
            x_end = min(w, center_x + bbox_width // 2 + 1)
            y_start = max(0, center_y - bbox_height // 2)
            y_end = min(h, center_y + bbox_height // 2 + 1)
            
            surrounding_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
            max_z = torch.max(surrounding_area)
            
            if max_z > 0:
                max_z_indices = torch.where(surrounding_area == max_z)
                y_offset, x_offset = max_z_indices[0][0], max_z_indices[1][0]
                center_x = x_start + x_offset
                center_y = y_start + y_offset
                center_z = max_z
            else:
                # center_z = float('nan')
                # center_z = 0.0
                center_z = torch.rand(1, device=device) * 60
        
        result.append(torch.tensor([cid, center_x, center_y, center_z], device=device))

    return torch.stack(result)

def find_rois_nonzero_z_adv(detections, depth_map): # perplexity 알고리즘
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    cam_indices = detections[:, 0].long()
    bboxes = detections[:, 1:]
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w - 1)
    y_min = torch.clamp(y_min, 0, h - 1)
    x_max = torch.clamp(x_max, 0, w - 1)
    y_max = torch.clamp(y_max, 0, h - 1)
    
    result = []
    fallback_counter = 0  # 카운터 초기화
    total_boxes = len(detections)  # 전체 박스 개수
    
    for i in range(len(detections)):
        cid = cam_indices[i]
        
        # 바운딩 박스 전체 영역 추출
        roi_depth = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        
        # 유효한 depth 값들만 선택
        valid_depths = roi_depth[roi_depth > 0]
        
        if len(valid_depths) > 0:
            # 유효한 depth 값들의 통계 계산
            max_z = torch.max(valid_depths)
            median_z = torch.median(valid_depths)
            
            # 최대 depth 위치 찾기
            max_z_pos = torch.where(roi_depth == max_z)
            local_y, local_x = max_z_pos[0][0], max_z_pos[1][0]
            
            # 전역 좌표로 변환
            center_x = x_min[i] + local_x
            center_y = y_min[i] + local_y
            
            # 최대값이 이상치일 수 있으므로, 중앙값과 비교하여 선택
            if max_z > 2.0 * median_z:
                center_z = median_z
            else:
                center_z = max_z
        else:
            # 바운딩 박스를 확장하여 주변 영역까지 탐색
            expand_ratio = 0.2  # 20% 확장
            bbox_w = x_max[i] - x_min[i]
            bbox_h = y_max[i] - y_min[i]
            
            x_start = max(0, x_min[i] - int(bbox_w * expand_ratio))
            x_end = min(w, x_max[i] + int(bbox_w * expand_ratio))
            y_start = max(0, y_min[i] - int(bbox_h * expand_ratio))
            y_end = min(h, y_max[i] + int(bbox_h * expand_ratio))
            
            expanded_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
            valid_expanded = expanded_area[expanded_area > 0]
            
            if len(valid_expanded) > 0:
                # 확장 영역에서 발견된 유효한 depth의 중앙값 사용
                center_z = torch.median(valid_expanded)
                # 원본 바운딩 박스의 중심점 사용
                center_x = (x_min[i] + x_max[i]) // 2
                center_y = (y_min[i] + y_max[i]) // 2
            else:
                # 마지막 수단으로 바운딩 박스 중심점과 예상 깊이값 사용
                fallback_counter += 1  # else 구문 실행 시 카운터 증가
                center_x = (x_min[i] + x_max[i]) // 2
                center_y = (y_min[i] + y_max[i]) // 2
                # 이미지 크기에 기반한 예상 깊이값 (가까운 거리 선호)
                center_z = torch.tensor(20.0, device=device)  # 기본 예상 깊이값
        
        result.append(torch.tensor([cid, center_x, center_y, center_z], device=device))
    
    # # 통계 출력
    # print(f"Total boxes: {total_boxes}, Fallback cases: {fallback_counter}, Ratio: {fallback_counter/total_boxes*100:.2f}%")

    return torch.stack(result)

def find_rois_nonzero_z_adv1(detections, depth_map):
    device = depth_map.device  # GPU 장치 사용 보장
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)  # 이미 float32이므로 형 변환 제거
    
    # 통계값 계산 부분
    cam_means = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    cam_medians = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    
    # 배치 평균 계산
    non_zero_mask = depth_map_re > 0
    batch_mean = torch.mean(depth_map_re[non_zero_mask]) if torch.any(non_zero_mask) else torch.tensor(0.0, device=device)
    
    # 카메라별 통계값 계산
    for cid in range(batch_size * num_cam):
        cam_depth = depth_map_re[cid]
        non_zero = cam_depth[cam_depth > 0]
        
        if len(non_zero) > 0:
            cam_means[cid] = torch.mean(non_zero)
            cam_medians[cid] = torch.median(non_zero)
        else:
            cam_means[cid] = batch_mean
            cam_medians[cid] = batch_mean

    # Detection 처리
    cam_indices = detections[:, 0].long().to(device)  # GPU로 이동
    bboxes = detections[:, 1:].to(device)  # GPU로 이동
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i]
        original_center_x = (x_min[i] + x_max[i]) // 2
        original_center_y = (y_min[i] + y_max[i]) // 2
        
        # 초기값 설정 (UnboundLocalError 방지)
        center_x, center_y = original_center_x, original_center_y
        center_z = depth_map_re[cid, original_center_y, original_center_x]
        
        if center_z == 0:
            # 2nd try: Full bounding box search
            bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            if bbox_area.numel() > 0:
                max_z = torch.max(bbox_area)
                if max_z > 0:
                    max_pos = torch.nonzero(bbox_area == max_z).float().mean(dim=0)
                    center_y = y_min[i] + int(max_pos[0])
                    center_x = x_min[i] + int(max_pos[1])
                    center_z = max_z
                    
            if center_z == 0:
                # 3rd try: Expanded search area
                bbox_width = x_max[i] - x_min[i] + 1
                bbox_height = y_max[i] - y_min[i] + 1
                x_start = max(0, original_center_x - bbox_width)
                x_end = min(w, original_center_x + bbox_width + 1)
                y_start = max(0, original_center_y - bbox_height)
                y_end = min(h, original_center_y + bbox_height + 1)
                
                expanded_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
                if expanded_area.numel() > 0:
                    max_z = torch.max(expanded_area)
                    if max_z > 0:
                        max_pos = torch.nonzero(expanded_area == max_z).float().mean(dim=0)
                        center_y = y_start + int(max_pos[0])
                        center_x = x_start + int(max_pos[1])
                        center_z = max_z
                        
        if center_z == 0:  # All attempts failed
            # Use camera median -> camera mean -> batch mean hierarchy
            center_z = cam_medians[cid] if cam_medians[cid] > 0 else cam_means[cid]
            center_x, center_y = original_center_x, original_center_y  # Keep original coordinates
        
        # 결과 텐서 생성 (GPU에서 수행)
        result.append(torch.tensor([cid, center_x, center_y, center_z], device=device))
    
    return torch.stack(result)

def find_rois_nonzero_z_adv2(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    # 통계값 계산 부분 (벡터화)
    non_zero_mask = depth_map_re > 0
    batch_mean = torch.mean(depth_map_re[non_zero_mask]) if torch.any(non_zero_mask) else torch.tensor(0.0, device=device)
    
    cam_means = torch.where(torch.sum(non_zero_mask, dim=(1,2)) > 0,
                            torch.sum(depth_map_re, dim=(1,2)) / torch.sum(non_zero_mask, dim=(1,2)),
                            batch_mean)
    
    # 중앙값 계산 (근사값 사용)
    k = torch.sum(non_zero_mask, dim=(1,2)) // 2
    cam_medians = torch.zeros_like(cam_means)
    for i in range(batch_size * num_cam):
        if k[i] > 0:
            cam_medians[i] = torch.kthvalue(depth_map_re[i].reshape(-1), k[i].item())[0]
        else:
            cam_medians[i] = batch_mean

    # Detection 처리 (벡터화)
    cam_indices = detections[:, 0].long().to(device)
    bboxes = detections[:, 1:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # 초기 중심점 깊이값
    center_z = depth_map_re[cam_indices, center_y, center_x]
    
    # 2nd try: Full bounding box search (벡터화)
    bbox_areas = [depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1] for i, cid in enumerate(cam_indices)]
    max_z_bbox = torch.stack([area.max() if area.numel() > 0 else torch.tensor(0.0, device=device) for area in bbox_areas])
    
    max_pos_bbox = []
    for area in bbox_areas:
        if area.numel() > 0:
            flat_area = area.reshape(-1)
            max_idx = flat_area.argmax()
            max_pos = torch.tensor([max_idx // area.size(1), max_idx % area.size(1)], device=device)
        else:
            max_pos = torch.tensor([0, 0], device=device)
        max_pos_bbox.append(max_pos)
    max_pos_bbox = torch.stack(max_pos_bbox)
    
    # 3rd try: Expanded search area (벡터화)
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    x_start = torch.clamp(center_x - bbox_width, 0, w-1)
    x_end = torch.clamp(center_x + bbox_width + 1, 0, w)
    y_start = torch.clamp(center_y - bbox_height, 0, h-1)
    y_end = torch.clamp(center_y + bbox_height + 1, 0, h)
    
    expanded_areas = [depth_map_re[cid, y_start[i]:y_end[i], x_start[i]:x_end[i]] for i, cid in enumerate(cam_indices)]
    max_z_expanded = torch.stack([area.max() if area.numel() > 0 else torch.tensor(0.0, device=device) for area in expanded_areas])
    
    max_pos_expanded = []
    for area in expanded_areas:
        if area.numel() > 0:
            flat_area = area.reshape(-1)
            max_idx = flat_area.argmax()
            max_pos = torch.tensor([max_idx // area.size(1), max_idx % area.size(1)], device=device)
        else:
            max_pos = torch.tensor([0, 0], device=device)
        max_pos_expanded.append(max_pos)
    max_pos_expanded = torch.stack(max_pos_expanded)
    
    # 결과 결정 (벡터화)
    center_z = torch.where(center_z > 0, center_z,
                           torch.where(max_z_bbox > 0, max_z_bbox,
                                       torch.where(max_z_expanded > 0, max_z_expanded,
                                                   torch.where(cam_medians[cam_indices] > 0, cam_medians[cam_indices], cam_means[cam_indices]))))
    
    center_x = torch.where(center_z == depth_map_re[cam_indices, center_y, center_x], center_x,
                           torch.where(center_z == max_z_bbox, x_min + max_pos_bbox[:, 1],
                                       torch.where(center_z == max_z_expanded, x_start + max_pos_expanded[:, 1],
                                                   center_x)))
    
    center_y = torch.where(center_z == depth_map_re[cam_indices, center_y, center_x], center_y,
                           torch.where(center_z == max_z_bbox, y_min + max_pos_bbox[:, 0],
                                       torch.where(center_z == max_z_expanded, y_start + max_pos_expanded[:, 0],
                                                   center_y)))
    
    result = torch.stack([cam_indices.float(), center_x.float(), center_y.float(), center_z], dim=1)
    
    return result

def find_rois_nonzero_z_adv3(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    # 통계값 및 신뢰도 스코어 초기화
    cam_means = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    cam_medians = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    confidence_scores = torch.zeros(len(detections), device=device, dtype=torch.float32)
    
    # 배치 평균 계산
    non_zero_mask = depth_map_re > 0
    batch_mean = torch.mean(depth_map_re[non_zero_mask]) if torch.any(non_zero_mask) else torch.tensor(0.0, device=device)
    
    # 카메라별 통계값 계산
    for cid in range(batch_size * num_cam):
        cam_depth = depth_map_re[cid]
        non_zero = cam_depth[cam_depth > 0]
        if len(non_zero) > 0:
            cam_means[cid] = torch.mean(non_zero)
            cam_medians[cid] = torch.median(non_zero)
        else:
            cam_means[cid] = batch_mean
            cam_medians[cid] = batch_mean

    # Detection 처리
    cam_indices = detections[:, 0].long().to(device)
    bboxes = detections[:, 1:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    def interpolate_depth(depth_slice, x, y, window_size=3):
        x1, x2 = max(0, x-window_size), min(w, x+window_size+1)
        y1, y2 = max(0, y-window_size), min(h, y+window_size+1)
        window = depth_slice[y1:y2, x1:x2]
        valid_depths = window[window > 0]
        return valid_depths.mean() if len(valid_depths) > 0 else 0
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i]
        bbox_width = x_max[i] - x_min[i] + 1
        bbox_height = y_max[i] - y_min[i] + 1
        
        # 동적 서치 윈도우 크기 계산
        search_radius_x = max(bbox_width // 4, 3)
        search_radius_y = max(bbox_height // 4, 3)
        
        original_center_x = (x_min[i] + x_max[i]) // 2
        original_center_y = (y_min[i] + y_max[i]) // 2
        center_x, center_y = original_center_x, original_center_y
        
        # 1단계: 중심점 깊이값 보간
        center_z = interpolate_depth(depth_map_re[cid], original_center_x, original_center_y)
        confidence_score = 1.0
        
        if center_z == 0:
            # 2단계: 동적 윈도우 기반 bbox 검색
            bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            if bbox_area.numel() > 0:
                # 깊이 값의 연속성을 고려한 가중치 맵 생성
                std_val = bbox_area.std()
                if std_val > 0:
                    diff = torch.abs(bbox_area - bbox_area.mean())
                    weight_factor = -diff / std_val
                    depth_weights = torch.exp(weight_factor)
                else:
                    depth_weights = torch.ones_like(bbox_area)
                weighted_depths = bbox_area * depth_weights
                max_z = torch.max(weighted_depths)
                if max_z > 0:
                    max_pos = torch.nonzero(weighted_depths == max_z).float().mean(dim=0)
                    center_y = y_min[i] + int(max_pos[0])
                    center_x = x_min[i] + int(max_pos[1])
                    center_z = max_z
                    confidence_score = 0.8
        
        if center_z == 0:
            # 3단계: 확장된 동적 서치
            x_start = max(0, original_center_x - search_radius_x)
            x_end = min(w, original_center_x + search_radius_x + 1)
            y_start = max(0, original_center_y - search_radius_y)
            y_end = min(h, original_center_y + search_radius_y + 1)
            
            expanded_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
            if expanded_area.numel() > 0:
                valid_depths = expanded_area[expanded_area > 0]
                if len(valid_depths) > 0:
                    center_z = torch.median(valid_depths)  # 중앙값 사용
                    valid_pos = torch.nonzero(expanded_area > 0).float().mean(dim=0)
                    center_y = y_start + int(valid_pos[0])
                    center_x = x_start + int(valid_pos[1])
                    confidence_score = 0.6
        
        if center_z == 0:
            # 최종 fallback: 통계값 사용
            if cam_medians[cid] > 0:
                center_z = cam_medians[cid]
                confidence_score = 0.4
            elif cam_means[cid] > 0:
                center_z = cam_means[cid]
                confidence_score = 0.2
            else:
                center_z = batch_mean
                confidence_score = 0.1
            center_x, center_y = original_center_x, original_center_y
        
        confidence_scores[i] = confidence_score
        result.append(torch.tensor([cid, center_x, center_y, center_z, confidence_score], device=device))
    
    return torch.stack(result), confidence_scores

def find_rois_nonzero_z_adv4(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    confidence_scores = torch.zeros(len(detections), device=device)
    
    cam_indices = detections[:, 0].long().to(device)
    bboxes = detections[:, 1:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        cx = (x_min[i] + x_max[i]) // 2
        cy = (y_min[i] + y_max[i]) // 2
        center_z = 0.0
        conf = 0.0
        
        # 1단계: 3x3 윈도우 내 최대 Z 값 검출
        y_start = max(0, cy-1)
        y_end = min(h, cy+2)
        x_start = max(0, cx-1)
        x_end = min(w, cx+2)
        
        window = depth_map_re[cid, y_start:y_end, x_start:x_end]
        valid_depths = window[window > 0]
        
        if valid_depths.numel() > 0:
            max_z = valid_depths.max()
            max_pos = (window == max_z).nonzero()[0]
            local_y = max_pos[0].item()
            local_x = max_pos[1].item()
            center_z = max_z.item()
            cx = x_start + local_x
            cy = y_start + local_y
            conf = 1.0

        # 2단계: BBox 내 최대 Z 값 검출
        if conf == 0.0:
            bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            valid_depths = bbox_area[bbox_area > 0]
            
            if valid_depths.numel() > 0:
                max_z = valid_depths.max()
                max_pos = (bbox_area == max_z).nonzero()[0]
                local_y = max_pos[0].item()
                local_x = max_pos[1].item()
                center_z = max_z.item()
                cx = x_min[i] + local_x
                cy = y_min[i] + local_y
                conf = 0.8

        # 3단계: 모델 예측값 사용
        if conf == 0.0:
            cx = int(model_pred_z[i,0].item() * w)
            cy = int(model_pred_z[i,1].item() * h)
            center_z = model_pred_z[i,2].item()
            conf = 0.7

        # 좌표 클램핑
        cx = torch.clamp(torch.tensor(cx), 0, w-1).item()
        cy = torch.clamp(torch.tensor(cy), 0, h-1).item()
        
        result.append(torch.tensor(
            [cid, cx, cy, center_z, conf], 
            device=device
        ))
        confidence_scores[i] = conf
    
    return torch.stack(result), confidence_scores

def find_rois_nonzero_z_adv5(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    cam_indices = detections[:, 0].long().to(device)
    bboxes = detections[:, 1:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        points = []
        conf = 0.0
        
        bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        nonzero_indices = (bbox_area > 0).nonzero()
        
        if nonzero_indices.size(0) > 0:
            # 실제 측정값이 있는 경우
            global_x = x_min[i] + nonzero_indices[:, 1].float()
            global_y = y_min[i] + nonzero_indices[:, 0].float()
            z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            conf = 1.0
            
            # 포인트 생성 [cam_id, x, y, z, confidence]
            points = torch.stack([
                torch.full_like(global_x, cid),
                global_x,
                global_y,
                z_values,
                torch.full_like(global_x, conf)
            ], dim=1)
            
        else:
            # 모델 예측값 사용
            cx = (model_pred_z[i,0] * w).clamp(0, w-1)
            cy = (model_pred_z[i,1] * h).clamp(0, h-1)
            cz = model_pred_z[i,2]
            conf = 0.7
            
            # 단일 포인트 생성
            points = torch.tensor([
                [cid, cx, cy, cz, conf]
            ], device=device)
            
        all_points.append(points)
    
    # 최종 출력 [N_points, 5]
    return torch.cat(all_points, dim=0)

def find_rois_nonzero_z_adv6(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    # 객체 ID 자동 생성 (0부터 순차적 할당)
    cam_indices = detections[:, 0].long().to(device)
    obj_indices = torch.arange(len(detections), device=device).long()  # [NEW] 자동 인덱싱
    bboxes = detections[:, 1:].to(device)  # 인덱스 조정 (객체 ID 컬럼 제외)
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    # 좌표 클램핑
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()  # 생성된 객체 ID 사용
        points = []
        conf = 0.0
        
        bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        nonzero_indices = (bbox_area > 0).nonzero()
        
        if nonzero_indices.size(0) > 0:
            global_x = x_min[i] + nonzero_indices[:, 1].float()
            global_y = y_min[i] + nonzero_indices[:, 0].float()
            z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            conf = 1.0
            
            points = torch.stack([
                torch.full_like(global_x, cid),
                torch.full_like(global_x, oid),
                global_x,
                global_y,
                z_values,
                torch.full_like(global_x, conf)
            ], dim=1)
            
        else:
            cx = (model_pred_z[i,0] * w).clamp(0, w-1)
            cy = (model_pred_z[i,1] * h).clamp(0, h-1)
            cz = model_pred_z[i,2]
            conf = 0.7
            
            points = torch.tensor([
                [cid,oid,cx, cy, cz, conf]
            ], device=device)
            
        all_points.append(points)
    
    return torch.cat(all_points, dim=0)

def find_rois_nonzero_z_adv7(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    cam_indices = detections[:, 0].long().to(device)
    obj_indices = torch.arange(len(detections), device=device).long()
    bboxes = detections[:, 1:].to(device)
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()
        
        bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        nonzero_indices = (bbox_area > 0).nonzero()
        
        if nonzero_indices.size(0) > 0:
            global_x = x_min[i] + nonzero_indices[:, 1].float()
            global_y = y_min[i] + nonzero_indices[:, 0].float()
            z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            conf = 1.0
            
            points = torch.stack([
                torch.full_like(global_x, cid),
                torch.full_like(global_x, oid),
                global_x,
                global_y,
                z_values,
                torch.full_like(global_x, conf)
            ], dim=1)
            
            all_points.append(points)
    
    return torch.cat(all_points, dim=0) if all_points else torch.empty((0, 6), device=device)


def image_to_lidar_global_modi(det_uvz, gt_KT):
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    for cid in range(6):
        img2lidar = inverse_gt_kt[cid]
        mask = (det_uvz[:, 0] == cid)
        if mask.any():
            detection_uvz = det_uvz[mask, 1:]
            normalize_uvz = torch.cat([detection_uvz[:, :2] * detection_uvz[:, 2:3], detection_uvz[:, 2:3]], dim=1).float()
            uvz_homogeneous = torch.cat([normalize_uvz, normalize_uvz.new_ones([normalize_uvz.shape[0], 1])], dim=1)
            xyz_global = torch.matmul(img2lidar[:3, :3], uvz_homogeneous[:, :3].T).T + img2lidar[:3, 3]
            list_xyz_global.append(xyz_global)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
    else:
        xyz_global_torch = torch.empty(0, 3, device=det_uvz.device)
    
    return xyz_global_torch

def image_to_lidar_global_modi(det_uvz, gt_KT):
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    list_indices = []  # New list to store indices
    for cid in range(6):
        img2lidar = inverse_gt_kt[cid]
        mask = (det_uvz[:, 0] == cid)
        if mask.any():
            detection_uvz = det_uvz[mask, 1:]
            normalize_uvz = torch.cat([detection_uvz[:, :2] * detection_uvz[:, 2:3], detection_uvz[:, 2:3]], dim=1).float()
            uvz_homogeneous = torch.cat([normalize_uvz, normalize_uvz.new_ones([normalize_uvz.shape[0], 1])], dim=1)
            xyz_global = torch.matmul(img2lidar[:3, :3], uvz_homogeneous[:, :3].T).T + img2lidar[:3, 3]
            list_xyz_global.append(xyz_global)
            list_indices.append(det_uvz[mask, 0])  # Store corresponding indices
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)  # Combine indices and add a dimension
        xyz_global_torch = torch.cat([indices_torch, xyz_global_torch], dim=1)  # Concatenate indices with xyz_global
    else:
        xyz_global_torch = torch.empty(0, 4, device=det_uvz.device)  # Adjust shape to [0, 4]
    
    return xyz_global_torch

def image_to_lidar_global_modi1(det_uvz, gt_KT):
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    list_indices = []
    list_confidence = []  # New list for confidence scores
    list_object_indices =[]
    
    for cid in range(6):
        img2lidar = inverse_gt_kt[cid]
        mask = (det_uvz[:, 0] == cid)
        if mask.any():
            detection_uvz = det_uvz[mask, 2:5]  # Only take x,y,z coordinates (excluding confidence)
            confidence = det_uvz[mask, 5]  # Get confidence scores
            ob_id = det_uvz[mask,1]
            id = det_uvz[mask,0]
            
            normalize_uvz = torch.cat([detection_uvz[:, :2] * detection_uvz[:, 2:3], detection_uvz[:, 2:3]], dim=1).float()
            uvz_homogeneous = torch.cat([normalize_uvz, normalize_uvz.new_ones([normalize_uvz.shape[0], 1])], dim=1)
            xyz_global = torch.matmul(img2lidar[:3, :3], uvz_homogeneous[:, :3].T).T + img2lidar[:3, 3]
            
            list_xyz_global.append(xyz_global)
            list_indices.append(id)
            list_object_indices.append(ob_id)
            list_confidence.append(confidence)  # Store confidence scores
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)
        confidence_torch = torch.cat(list_confidence, dim=0).unsqueeze(1)  # Add confidence scores
        ob_id_torch = torch.cat(list_object_indices, dim=0).unsqueeze(1)
        
        # Concatenate indices, xyz_global, and confidence scores
        xyz_global_torch = torch.cat([indices_torch, ob_id_torch, xyz_global_torch, confidence_torch], dim=1)
    else:
        xyz_global_torch = torch.empty(0, 6, device=det_uvz.device)  # Adjust shape to [0, 5]
    
    return xyz_global_torch

def lidar_to_image_with_index(det_xyz, gt_KT, img_shape=(512,1408)):
    """
    Args:
        det_xyz: [N, 4] (camera_index, x, y, z)
        gt_KT: [6, 4, 4] (카메라별 변환 행렬)
        img_shape: (H, W) 이미지 해상도 (세로, 가로)
    
    Returns:
        uvz_global_torch: [M, 4] (유효한 포인트만 포함)
        mask_valid_global: [N] (전체 포인트에 대한 유효성 마스크)
    """
    list_uvz_global = []
    list_indices = []
    mask_valid_global = torch.zeros(det_xyz.shape[0], dtype=torch.bool, device=det_xyz.device)

    for cid in range(6):
        lidar2img = gt_KT[cid]
        mask = (det_xyz[:, 0] == cid)  # 현재 카메라 ID에 해당하는 포인트 필터링
        if mask.any():
            detection_xyz = det_xyz[mask, 1:]  # x,y,z 좌표 추출
            points_lidar2img = (lidar2img @ detection_xyz.T).T  # [N,3]
            points_lidar2img = torch.cat([points_lidar2img[:, :2] / points_lidar2img[:, 2:3], points_lidar2img[:, 2:3]], dim=1)  # [N,3] (u,v,z)

            # 유효한 포인트 필터링
            pcl_uv = points_lidar2img[:, :2]  # u,v 좌표
            pcl_z = points_lidar2img[:, 2]   # 깊이 값
            mask_valid = (
                (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) &  # u 범위 체크 (0 < u < W)
                (pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < img_shape[0]) &  # v 범위 체크 (0 < v < H)
                (pcl_z > 0)                                           # 깊이 값이 양수인지 확인
            )
            
            # 유효한 포인트만 저장
            valid_points = points_lidar2img[mask_valid]
            valid_indices = det_xyz[mask][mask_valid][:, 0]

            list_uvz_global.append(valid_points)
            list_indices.append(valid_indices)

            # 전체 마스크 업데이트
            mask_valid_global[mask.nonzero(as_tuple=True)[0]] = mask_valid

    if list_uvz_global:
        uvz_global_torch = torch.cat(list_uvz_global, dim=0)  # 유효한 포인트 병합
        indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)  # 인덱스 병합
        
        # 인덱스와 uvz 좌표 결합
        uvz_global_torch = torch.cat([indices_torch, uvz_global_torch], dim=1)  # [M,4]
    else:
        uvz_global_torch = torch.empty(0, 4, device=det_xyz.device)  # 빈 텐서 반환
    
    return uvz_global_torch, mask_valid_global

def lidar_to_image_no_filter(det_xyz, gt_KT):
    """
    Args:
        det_xyz: [N, 4] (camera_index, x, y, z)
        gt_KT: [6, 4, 4] (카메라별 변환 행렬)
    
    Returns:
        uvz_global_torch: [M, 4] (모든 포인트 포함)
    """
    list_uvz_global = []
    list_indices = []

    for cid in range(6):
        lidar2img = gt_KT[cid]
        mask = (det_xyz[:, 0] == cid)  # 현재 카메라 ID에 해당하는 포인트 필터링
        if mask.any():
            detection_xyz = det_xyz[mask, 1:]  # x,y,z 좌표 추출
            points_lidar2img = (lidar2img @ detection_xyz.T).T  # [N,3]
            points_lidar2img = torch.cat([points_lidar2img[:, :2] / points_lidar2img[:, 2:3], points_lidar2img[:, 2:3]], dim=1)  # [N,3] (u,v,z)

            list_uvz_global.append(points_lidar2img)
            list_indices.append(det_xyz[mask][:, 0])

    if list_uvz_global:
        uvz_global_torch = torch.cat(list_uvz_global, dim=0)  # 모든 포인트 병합
        indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)  # 인덱스 병합
        
        # 인덱스와 uvz 좌표 결합
        uvz_global_torch = torch.cat([indices_torch, uvz_global_torch], dim=1)  # [M,4]
    else:
        uvz_global_torch = torch.empty(0, 4, device=det_xyz.device)  # 빈 텐서 반환
    
    return uvz_global_torch

 
def project_lidar_to_image(pts_hom: torch.Tensor, 
                           lidar2img: torch.Tensor, 
                           eps: float = 1e-6) -> torch.Tensor:
    """
    LiDAR 포인트 클라우드를 이미지 평면에 투영 (배치 처리 지원)

    Args:
        pts_hom (torch.Tensor): 동차 좌표계 라이다 포인트 [B, 4] (x, y, z, 1)
        lidar2img (torch.Tensor): 변환 행렬 [B, 4, 4]
        eps (float): 0으로 나누기 방지용 작은 값 (기본값: 1e-6)

    Returns:
        torch.Tensor: 이미지 평면 UV 좌표와 깊이 [B, 3] (u, v, z)

    Raises:
        ValueError: 입력 차원이 유효하지 않은 경우
    """
    # 차원 검증
    if pts_hom.dim() != 2 or pts_hom.size(1) != 4:
        raise ValueError(f"pts_hom은 [B,4] 형태여야 합니다. 현재 형태: {pts_hom.shape}")
    if lidar2img.dim() != 3 or lidar2img.size(1) != 4 or lidar2img.size(2) != 4:
        raise ValueError(f"lidar2img은 [B,4,4] 형태여야 합니다. 현재 형태: {lidar2img.shape}")
    if pts_hom.size(0) != lidar2img.size(0):
        raise ValueError(f"배치 크기 불일치: pts_hom={pts_hom.size(0)}, lidar2img={lidar2img.size(0)}")

    # 차원 확장 및 변환
    lidar2img = lidar2img.float()
    pts_hom_ = pts_hom.unsqueeze(-1)  # [B,4,1]
    cam_points = torch.bmm(lidar2img, pts_hom_).squeeze(-1)  # [B,4]

    # 좌표 정규화
    u_coords = cam_points[:, 0] / (cam_points[:, 2] + eps)
    v_coords = cam_points[:, 1] / (cam_points[:, 2] + eps)
    z_depth = cam_points[:, 2]

    return torch.stack([u_coords, v_coords, z_depth], dim=1)  # [B,3]

def image_to_lidar_global(pred_uvz, gt_KT):
    """
    이미지 좌표(u, v, z)를 라이다 전역 좌표계로 변환
    Args:
        pred_uvz: [B, N, 3] (u, v, z) 이미지 좌표
        gt_KT: [B, 4, 4] 이미지->라이다 변환 행렬
    Returns:
        xyz_global: [B, N, 3] 라이다 전역 좌표
    """
    B, N, _ = pred_uvz.shape
    device = pred_uvz.device
    
    # 역변환 행렬 계산 [B,4,4]
    inverse_gt_kt = torch.inverse(gt_KT)
    
    # 정규화 좌표 계산 (u*z, v*z, z)
    uvz_scaled = pred_uvz.clone()
    uvz_scaled[..., :2] *= uvz_scaled[..., 2:3]  # [u*z, v*z, z]
    
    # 동차 좌표 변환 [B, N, 4]
    uvz_homogeneous = torch.cat([
        uvz_scaled, 
        torch.ones(B, N, 1, device=device)
    ], dim=-1)
    
    # 회전 변환: [B,3,3] @ [B,3,N] → [B,3,N]
    rotation = inverse_gt_kt[:, :3, :3]
    translation = inverse_gt_kt[:, :3, 3]
    
    # 행렬 곱 수행 (배치 처리)
    xyz_rot = torch.matmul(
        rotation, 
        uvz_homogeneous[..., :3].permute(0, 2, 1)
    )  # [B,3,N]
    
    # 병렬 이동 적용 및 차원 재정렬
    xyz_lidar = (xyz_rot.permute(0, 2, 1) + translation.unsqueeze(1))
    
    return xyz_lidar

def selected_image_to_lidar_global(pred_uvz, gt_KT):
    """
    이미지 좌표(u, v, z) + 카메라 인덱스 → 라이다 전역 좌표 변환
    Args:
        pred_uvz: [B, N, 4] (cam_idx, u, v, z)
        gt_KT: [Total_Cams, 4, 4] 전체 카메라 변환 행렬 풀
    Returns:
        xyz_global: [B, N, 3] 라이다 좌표
    """
    device = pred_uvz.device
    B, N, _ = pred_uvz.shape
    
    # 1. 데이터 구조 재구성
    flat_uvz = pred_uvz.view(-1, 4)          # [B*N, 4]
    camera_indices = flat_uvz[:, 0].long()   # [B*N]
    
    # 2. 각 포인트에 해당하는 KT 행렬 선택
    selected_KT = gt_KT[camera_indices]      # [B*N, 4, 4]
    
    # 3. 역변환 행렬 계산
    inverse_KT = torch.inverse(selected_KT)  # [B*N, 4, 4]
    
    # 4. 좌표 변환 수행
    uvz_scaled = flat_uvz[:, 1:].clone()     # [B*N, 3]
    uvz_scaled[:, :2] *= uvz_scaled[:, 2:]   # u*z, v*z
    
    # 5. 동차 좌표 변환
    uvz_homo = torch.cat([
        uvz_scaled,
        torch.ones((B*N, 1), device=device)
    ], dim=1)                                # [B*N, 4]
    
    # 6. 행렬 연산 (배치 처리)
    xyz_rot = torch.bmm(
        inverse_KT[:, :3, :3],               # [B*N, 3, 3]
        uvz_homo[:, :3].unsqueeze(-1)        # [B*N, 3, 1]
    ).squeeze(-1)                            # [B*N, 3]
    
    xyz_global = xyz_rot + inverse_KT[:, :3, 3]
    
    # 7. 원본 배치 형태 복원
    return xyz_global.view(B, N, 3)

def miscalib_transform(det_xyz, mis_T):
    # inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    for cid in range(6):
        rotate_lidar2lidar = mis_T[cid]
        mask = (det_xyz[:, 0] == cid)
        if mask.any():
            detection_xyz = det_xyz[mask, 1:]
            rotated_points = detection_xyz[:, :3].matmul(rotate_lidar2lidar[:3, :3].T) + rotate_lidar2lidar[:3, 3].unsqueeze(0)
            # points_img_mis_calibrated = torch.cat([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]], 1)
            list_xyz_global.append(rotated_points)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
    else:
        xyz_global_torch = torch.empty(0, 3, device=det_xyz.device)
    
    return xyz_global_torch

def miscalib_transform1(det_xyz, mis_extrinsic):
    list_xyz_global = []
    list_confidence = []
    list_cam_indices = []  # New list for camera indices
    
    for cid in range(6):
        RT = mis_extrinsic[cid]
        # 회전 행렬(R)과 이동 벡터(t) 분리
        # R = RT[:3, :3]  # 회전 행렬 전치 (LiDAR → 카메라 좌표계 변환)
        # t = RT[:3, 3]
        
        mask = (det_xyz[:, 0] == cid)
        if mask.any():
            detection_xyz = det_xyz[mask, 1:4]  # Only take x,y,z coordinates
            confidence = det_xyz[mask, 4]  # Get confidence scores
            cam_indices = det_xyz[mask, 0]  # Get camera indices

            # 회전 적용: (N,3) @ (3,3) → (N,3)
            # rotated_points = detection_xyz @ R
            points_hom = torch.cat([detection_xyz, torch.ones_like(detection_xyz[:, :1])], dim=1)
            # Apply inverse transformation: LiDAR → Camera
            rotated_points = (RT @ points_hom.T).T
            rotated_points = rotated_points[:,:3]
            # 이동 적용: (N,3) + (1,3)
            # translated_points = rotated_points + t.unsqueeze(0)
            # # Z축 반전 (카메라 좌표계 방향 보정)
            # translated_points[:, 2] *= -1
            
            list_xyz_global.append(rotated_points)
            list_confidence.append(confidence)
            list_cam_indices.append(cam_indices)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        confidence_torch = torch.cat(list_confidence, dim=0).unsqueeze(1)
        cam_indices_torch = torch.cat(list_cam_indices, dim=0).unsqueeze(1)
        
        # Concatenate camera indices, rotated points, and confidence scores
        xyz_global_torch = torch.cat([cam_indices_torch, xyz_global_torch, confidence_torch], dim=1)
    else:
        xyz_global_torch = torch.empty(0, 5, device=det_xyz.device)  # [0, 5] for cam_id,x,y,z,confidence
    
    return xyz_global_torch

def miscalib_transform2(det_xyz, mis_Rt):
    list_xyz_global = []
    list_confidence = []
    list_cam_indices = []  
    list_ob_indices = []
    
    for cid in range(6):
        Rt_perturb = mis_Rt[cid]
        mask = (det_xyz[:, 0] == cid)
        if mask.any():
            detection_xyz = det_xyz[mask, 2:5]  
            confidence = det_xyz[mask, 5]  
            cam_indices = det_xyz[mask, 0]
            ob_indices = det_xyz[mask, 1]

            # Convert to homogeneous coordinates and apply inverse
            points_hom = torch.cat([
                detection_xyz, 
                torch.ones_like(detection_xyz[:, :1])
            ], dim=1)
            
            # LiDAR → LiDAR
            perturbed_points = (Rt_perturb @ points_hom.T).T[:,:3]
            # rotated = detection_xyz @ RT_perturb[:3,:3].T
            # disturbed_xyz = rotated + RT_perturb[:3, 3]
            
            list_xyz_global.append(perturbed_points)
            list_confidence.append(confidence)
            list_cam_indices.append(cam_indices)
            list_ob_indices.append(ob_indices)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        confidence_torch = torch.cat(list_confidence, dim=0).unsqueeze(1)
        cam_indices_torch = torch.cat(list_cam_indices, dim=0).unsqueeze(1)
        ob_indices_torch = torch.cat(list_ob_indices, dim=0).unsqueeze(1)
        
        xyz_global_torch = torch.cat(
            [cam_indices_torch,ob_indices_torch, xyz_global_torch, confidence_torch], 
            dim=1
        )
    else:
        xyz_global_torch = torch.empty(0, 6, device=det_xyz.device)
    
    return xyz_global_torch

def display_nonzero_depthmap(result_list, original_image):
    
    fig, axes = plt.subplots(1, len(result_list), figsize=(20, 5))
    if len(result_list) == 1:
        axes = [axes]
    
    for cam_idx, camera_results in enumerate(result_list):
        ax = axes[cam_idx]
        ax.imshow(original_image[cam_idx].permute(1,2,0).cpu().numpy())
        ax.set_title(f'Camera {cam_idx + 1}')
        
        for bbox_points in camera_results:
            if len(bbox_points) > 0:
                x = bbox_points[:, 0].cpu().numpy()
                y = bbox_points[:, 1].cpu().numpy()
                z = bbox_points[:, 2].cpu().numpy()
                
                scatter = ax.scatter(x, y, c=z, cmap='viridis', s=0.5, alpha=0.5)
        
        ax.set_xlim(0, original_image.shape[3])
        ax.set_ylim(original_image.shape[2], 0)
        ax.axis('off')
    
    plt.colorbar(scatter, ax=axes[-1], label='Depth')
    plt.tight_layout()
    plt.savefig('detection_bbox.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def normalize_point_cloud(point_cloud):
    # 포인트 클라우드의 형태: [batch_size, num_points, 3]
    
    # 1. 중심 이동: 각 포인트 클라우드의 평균을 계산하고 빼줍니다
    centroid = torch.mean(point_cloud, dim=1, keepdim=True)
    point_cloud = point_cloud - centroid

    # 2. 스케일 정규화: 원점으로부터의 최대 거리를 계산합니다
    max_distance = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=-1)), dim=1, keepdim=True)[0]
    
    # 3. 포인트 클라우드를 최대 거리로 나누어 [-1, 1] 범위로 정규화합니다
    normalized_point_cloud = point_cloud / max_distance.unsqueeze(-1)

    return normalized_point_cloud

def minmax_normalize_uvz(uvz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    UVZ 좌표를 채널별 민맥스 정규화 (0~1 범위)

    Args:
        uvz (torch.Tensor): 원본 UVZ 좌표 [B, 3]
        eps (float): 0 나누기 방지용 작은 값 (기본값: 1e-6)

    Returns:
        torch.Tensor: 정규화된 UVZ 좌표 [B, 3]
    """
    if uvz.dim() != 2 or uvz.size(1) != 3:
        raise ValueError(f"uvz는 [B,3] 형태여야 합니다. 현재 형태: {uvz.shape}")

    # 채널별 최소/최대 계산
    min_vals = uvz.min(dim=0, keepdim=True)[0]  # [1,3]
    max_vals = uvz.max(dim=0, keepdim=True)[0]  # [1,3]
    
    # 정규화
    uvz_norm = (uvz - min_vals) / (max_vals - min_vals + eps)
    return uvz_norm , min_vals , max_vals

def minmax_denormalize_uvz(uvz_norm: torch.Tensor, 
                          min_vals: torch.Tensor, 
                          max_vals: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    정규화된 UVZ 좌표를 원본 스케일로 역변환

    Args:
        uvz_norm (torch.Tensor): 정규화된 UVZ [B,3]
        min_vals (torch.Tensor): 원본 최솟값 [1,3] or [3]
        max_vals (torch.Tensor): 원본 최댓값 [1,3] or [3]
        eps (float): 수치 안정성용 작은 값 (기본값: 1e-6)

    Returns:
        torch.Tensor: 역정규화된 UVZ 좌표 [B,3]
    """
    # 차원 검증
    if uvz_norm.dim() != 2 or uvz_norm.size(1) != 3:
        raise ValueError(f"uvz_norm은 [B,3] 형태여야 합니다. 현재: {uvz_norm.shape}")
    if min_vals.shape != max_vals.shape:
        raise ValueError(f"min/max 차원 불일치: min={min_vals.shape}, max={max_vals.shape}")
    if min_vals.numel() != 3 or max_vals.numel() != 3:
        raise ValueError(f"min/max는 3개 채널 값을 가져야 합니다. 현재: min={min_vals.numel()}, max={max_vals.numel()}")

    # 차원 조정 (벡터 → 행렬)
    min_vals = min_vals.view(1, -1)  # [3] → [1,3]
    max_vals = max_vals.view(1, -1)  # [3] → [1,3]

    # 역정규화 계산
    range_vals = max_vals - min_vals + eps
    uvz_orig = uvz_norm * range_vals + min_vals
    
    return uvz_orig

def corrs_normalization(corrs,origin_img_shape=(900,1600,3),):
    
    # corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:,:, 0] = corrs[:,:, 0]/origin_img_shape[1] 
    # corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:,:, 1] = corrs[:,:, 1]/origin_img_shape[0] 
    if corrs[:,:, 2].numel() > 0:
        corrs[:,:, 2] = (corrs[:,:, 2]-torch.min(corrs[:,:, 2]))/(torch.max(corrs[:,:, 2]) - torch.min(corrs[:,:, 2]))
    else :
        corrs[:,:, 2] = (corrs[:,:, 2]-0)/(60 - 0)
    # corrs[:, 3] = (0.5*corrs[:, 3])/1280 + 0.5
    corrs[:,:, 3] = corrs[:,:, 3]/origin_img_shape[1]         
    # corrs[:, 4] = (0.5*corrs[:, 4])/384
    corrs[:,:, 4] = corrs[:,:, 4]/origin_img_shape[0]
    if corrs[:,:, 5].numel() > 0:
        corrs[:,:, 5] = (corrs[:,:, 5]-torch.min(corrs[:,:, 5]))/(torch.max(corrs[:,:, 5]) - torch.min(corrs[:,:, 5])) 
    else :
        corrs[:,:, 5] = (corrs[:,:, 5]-0)/(60 - 0)
    
    return corrs

def corrs_denormalization(corrs_modi, origin_img_shape=(900, 1600, 3)):
    # x 좌표 역정규화
    corrs = corrs_modi.clone().detach()
    corrs[:, :, 0] = corrs[:, :, 0] * origin_img_shape[1]
    
    # y 좌표 역정규화
    corrs[:, :, 1] = corrs[:, :, 1] * origin_img_shape[0]
    
    # z 좌표 역정규화
    if corrs[:, :, 2].numel() > 0:
        min_z = torch.min(corrs[:, :, 2])
        max_z = torch.max(corrs[:, :, 2])
        corrs[:, :, 2] = corrs[:, :, 2] * (max_z - min_z) + min_z
    else:
        corrs[:, :, 2] = corrs[:, :, 2] * 60
       
    return corrs

def points2depthmap(points, height, width ,downsample=1): 
    device = points.device
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 0.5], # original
        # 'depth': [1.0, 80.0, 0.5],
    }
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    coor = torch.round(points[:, :2] / downsample)
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height) & (
            depth < grid_config['depth'][1]) & (
                depth >= grid_config['depth'][0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth

    # 최종 유효 인덱스 계산
    valid_indices = torch.where(kept1)[0][kept2]
    
    return depth_map, coor, depth , valid_indices

def dense_map_gpu_optimized(Pts, n, m, grid):
    device = Pts.device
    ng = 2 * grid + 1
    epsilon = 1e-8  # 작은 값 추가하여 0으로 나누는 상황 방지
    # import pdb; pdb.set_trace()
    # 초기 텐서를 GPU로 이동
    mX = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mY = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mD = torch.zeros((m, n), dtype=torch.float32, device=device)

    mX_idx = Pts[1].clone().detach().to(dtype=torch.int64, device=device)
    mY_idx = Pts[0].clone().detach().to(dtype=torch.int64, device=device)

    mX[mX_idx, mY_idx] = Pts[0] - torch.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - torch.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)

    # KmX = torch.zeros((ng, ng), dtype=torch.float32, device=device)
    # KmY = torch.zeros((ng, ng), dtype=torch.float32, device=device)
    # KmD = torch.zeros((ng, ng), dtype=torch.float32, device=device)

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + j
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]

    S = torch.zeros_like(KmD[0, 0], device=device)
    Y = torch.zeros_like(KmD[0, 0], device=device)

    for i in range(ng):
        for j in range(ng):
            # s = 1 / torch.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2)
            s = 1 / torch.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2 + epsilon)
            Y += s * KmD[i, j]
            S += s

    S[S == 0] = 1
    out = torch.zeros((m, n), dtype=torch.float32, device=device)
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    # return out.cpu()  # 최종 결과를 CPU로 이동
    return out # 최종 결과를 GPU

def colormap(disp):
    """"Color mapping for disp -- [H, W] -> [3, H, W]"""
    disp_np = disp.cpu().numpy()        # tensor -> numpy
    # disp_np = disp
    # vmax = np.percentile(disp_np, 95)
    vmin = disp_np.min()
    vmax = disp_np.max()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    # return colormapped_im.transpose(2, 0, 1)
    # colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32)
    colormapped_tensor = torch.from_numpy(colormapped_im)
    return colormapped_tensor

def denormalize_points(normal_points, z_min=None, z_max=None):
    """
    정규화된 UVZ 좌표를 원본 좌표계로 변환
    Args:
        normal_points: [N,3] 정규화된 텐서 (u, v, z)
        z_min: 원본 z 최소값 (없을 경우 기본값 0 사용)
        z_max: 원본 z 최대값 (없을 경우 기본값 80 사용)
    Returns:
        denorm_points: [N,3] 역정규화된 텐서
    """
    denorm_points = normal_points.clone()
    
    # U 좌표 복원: [0.5~1.5) → [0~640)
    denorm_points[:,:, 0] = (denorm_points[:,:, 0] - 0.5) * 1280
    
    # V 좌표 복원: [0~1) → [0~192)
    denorm_points[:,:, 1] = denorm_points[:,:, 1] * 192
    
    # Z 좌표 복원
    if z_min is not None and z_max is not None:
        z_range = z_max - z_min
        denorm_points[:,:, 2] = denorm_points[:,:, 2] * z_range + z_min
    else:  # 기본 범위 사용 (0~80)
        denorm_points[:,:, 2] = denorm_points[:,:, 2] * 80
        
    return denorm_points

def pixel_to_normalized(uv_pixel, intrinsics):
    """ 배치 처리된 픽셀 → 정규화 좌표 변환 """
    # uv_pixel: [B, N, 5] (batch, num_points, [cam, obj, u, v, z])
    # intrinsics: [O,4,4] (객체별 내부 파라미터)
    
    # 1. 배치 차원 병합
    B, N, _ = uv_pixel.shape
    uv_flat = uv_pixel.view(-1, 5)  # [B*N, 5]
    
    # 2. 객체 ID 기반 파라미터 선택
    obj_ids = uv_flat[:, 1].long()  # [B*N]
    obj_intrinsics = intrinsics[obj_ids]  # [B*N,4,4]
    
    # 3. 파라미터 추출
    fx = obj_intrinsics[:, 0, 0]  # [B*N]
    fy = obj_intrinsics[:, 1, 1]  # [B*N]
    cx = obj_intrinsics[:, 0, 2]  # [B*N]
    cy = obj_intrinsics[:, 1, 2]  # [B*N]
    
    # 4. 좌표 분해 및 계산
    u = uv_flat[:, 2]  # [B*N]
    v = uv_flat[:, 3]  # [B*N]
    z = -uv_flat[:, 4]  # [B*N]
    
    u_norm = (u - cx) / fx
    v_norm = (v - cy) / fy
    
    # 5. 배치 차원 복원
    normalized = torch.stack([
        uv_flat[:, 0],  # cam
        uv_flat[:, 1],  # obj
        u_norm,
        v_norm,
        z
    ], dim=1).view(B, N, 5)  # [B, N, 5]
    
    return normalized

def center2lidar_batch(center_pred, intrinsics, extrinsics):
    """
    center_pred: [B, N, 5] (cam_index, obj_id, u, v, z)
    intrinsics: [O,4,4] (객체별 내부 파라미터)
    extrinsics: [O,4,4] (객체별 외부 파라미터)
    """
    B, N, _ = center_pred.shape
    device = center_pred.device
    
    # 1. 객체 ID 추출 및 파라미터 선택
    obj_ids = center_pred[:, :, 1].long()  # [B, N]
    obj_ids_flat = obj_ids.view(-1)  # [B*N]
    
    # 객체별 파라미터 선택 및 차원 조정
    selected_intrinsics = intrinsics[obj_ids_flat].view(B, N, 4, 4)  # [B, N, 4, 4]
    selected_extrinsics = extrinsics[obj_ids_flat].view(B, N, 4, 4)  # [B, N, 4, 4]

    # 2. 좌표 변환 준비
    u = center_pred[:, :, 2]  # [B, N]
    v = center_pred[:, :, 3]  # [B, N]
    z = center_pred[:, :, 4]  # [B, N]
    
    # 3. 이미지 좌표계 → 라이다 좌표계 변환
    center_img = torch.stack([
        u * z,
        v * z,
        z,
        torch.ones_like(z)
    ], dim=-1)  # [B, N, 4]

    # 4. 변환 행렬 계산
    lidar2img = torch.matmul(
        selected_intrinsics,
        selected_extrinsics.transpose(2, 3)  # [B, N, 4, 4]
    )
    
    # 5. 역변환 행렬 계산
    img2lidar = torch.inverse(lidar2img)  # [B, N, 4, 4]

    # 6. 좌표 변환 수행
    center_lidar = torch.matmul(
        img2lidar,
        center_img.unsqueeze(-1)  # [B, N, 4, 1]
    ).squeeze(-1)[:, :, :3]  # [B, N, 3]

    center_lidar_with_index = torch.cat([center_pred[...,0:2],center_lidar],dim=2)

    return center_lidar_with_index,center_lidar, lidar2img


def draw_correspondences(trimed_corrs, sbs_img, camera_idx=0, save_path='correspond.jpg'):
    """정규화 좌표 기반 시각화 (0~1 범위 입력 필요)"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 1. 이미지 전처리
    img_tensor = sbs_img[camera_idx]  # [3, 192, 1280]
    denorm_img = img_tensor / 2 + 0.5  # 정규화 해제
    img_np = denorm_img.permute(1, 2, 0).cpu().numpy()
    
    # 2. 좌표 추출 및 스케일 복원
    H, W = img_np.shape[:2]
    left_pts = trimed_corrs[:, :2].detach().cpu().numpy()  # 정규화 좌표 [N,2] (0~1)
    right_pts = trimed_corrs[:, 3:5].detach().cpu().numpy()
    
    # 3. 정규화 → 픽셀 좌표 변환 (원본 알고리즘 반영)
    left_pts[:, 0] = left_pts[:, 0] * 640  # u = (norm_u - 0.5)*640 
    left_pts[:, 1] = left_pts[:, 1] * 192  # v = norm_v * 192
    right_pts[:, 0] = (right_pts[:, 0] - 0.5) * 640 + 640  # 우측 오프셋 적용
    right_pts[:, 1] = right_pts[:, 1] * 192

    # 4. 좌표 클리핑 및 필터링
    left_pts[:, 0] = np.clip(left_pts[:, 0], 0, W-1)
    left_pts[:, 1] = np.clip(left_pts[:, 1], 0, H-1)
    right_pts[:, 0] = np.clip(right_pts[:, 0], 0, W-1)
    right_pts[:, 1] = np.clip(right_pts[:, 1], 0, H-1)
    
    valid_mask = ~(np.isnan(left_pts).any(axis=1) | np.isnan(right_pts).any(axis=1))
    left_pts = left_pts[valid_mask]
    right_pts = right_pts[valid_mask]

    # 5. 시각화
    plt.figure(figsize=(20, 6))
    plt.imshow(img_np)
    
    # 좌측 포인트 (청색)
    plt.scatter(left_pts[:,0], left_pts[:,1], 
                c='cyan', s=30, edgecolors='k', linewidth=0.8, label='Left Points')
    # 우측 포인트 (자홍색)
    plt.scatter(right_pts[:,0], right_pts[:,1], 
                c='magenta', s=30, edgecolors='k', linewidth=0.8, label='Right Points')
    
    # 연결선 그리기 (옵션)
    for l, r in zip(left_pts, right_pts):
        plt.plot([l[0], r[0]], [l[1], r[1]], 
                color='yellow', linestyle='--', linewidth=1.5, alpha=0.4)

    plt.axis('off')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correspondence image saved to {save_path}")



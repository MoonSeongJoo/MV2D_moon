import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch.nn.functional as F
import torchvision

def display_depth_maps(imgs, depth_map, mis_calibrated_depth_map):
    # """
    # 이미지 위에 depth map과 mis_calibrated depth map을 오버레이하여 디스플레이하는 함수
    # """

    for cid in range(imgs.shape[0]):

        img_np = imgs.squeeze()[cid].cpu().detach().numpy()
        depth_gt_np = depth_map.squeeze()[cid].cpu().detach().numpy() 
        depth_mis_np = mis_calibrated_depth_map.squeeze()[cid].cpu().detach().numpy()
        # sbs_img_np = sbs_img.squeeze()[cid].cpu().detach().numpy()
        img_np = np.transpose(img_np,(1,2,0))
        depth_gt_np = np.transpose(depth_gt_np,(1,2,0))
        depth_mis_np = np.transpose(depth_mis_np,(1,2,0))
        # sbs_img_np = np.transpose(sbs_img_np,(1,2,0))
        # 이미지 데이터가 float 타입인 경우 0과 1 사이로 정규화
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            depth_gt_np = (depth_gt_np - depth_gt_np.min()) / (depth_gt_np.max() - depth_gt_np.min())
            depth_mis_np = (depth_mis_np - depth_mis_np.min()) / (depth_mis_np.max() - depth_mis_np.min())
            # sbs_img_np = (sbs_img_np - sbs_img_np.min()) / (sbs_img_np.max() - sbs_img_np.min())

        # input display
        ####### display input signal #########        
        plt.figure(figsize=(20, 20))
        plt.subplot(311)
        plt.imshow(img_np)
        plt.title("camera_input", fontsize=15)
        plt.axis('off')

        plt.subplot(312)
        plt.imshow( depth_gt_np, cmap='magma')
        plt.title("calibrated_lidar_input", fontsize=15)
        plt.axis('off') 

        plt.subplot(313)
        plt.imshow( depth_mis_np, cmap='magma')
        plt.title("mis-calibrated_lidar_input", fontsize=15)
        plt.axis('off')

        # plt.subplot(224)
        # plt.imshow( sbs_img_np)
        # plt.title("sbs_img_input", fontsize=15)
        # plt.axis('off')
        
        plt.tight_layout(pad=0)  # 여백 제거
        plt.savefig(f'raw_input_image_{cid+1}.png', dpi=300, bbox_inches='tight')
        plt.close('all')
    
    print ('display end')
    # ############ end of display input signal ###################

def visualize_depth_maps(lidar_depth_map_gt, lidar_depth_map_mis):
    num_cameras = lidar_depth_map_gt.shape[0]
    fig, axes = plt.subplots(num_cameras, 2, figsize=(12, 6*num_cameras))
    
    for i in range(num_cameras):
        # Ground Truth
        gt = lidar_depth_map_gt[i].cpu().numpy() if torch.is_tensor(lidar_depth_map_gt) else lidar_depth_map_gt[i]
        axes[i,0].imshow(gt, cmap='viridis', vmin=0, vmax=80)  # 거리 범위 0-80m 가정
        
        # Miscalibrated
        mis = lidar_depth_map_mis[i]
        if mis.ndim == 3:  # (C,H,W) → (H,W,C)
            mis = np.transpose(mis, (1,2,0))
        mis = (mis - np.min(mis)) / (np.max(mis) - np.min(mis) + 1e-8)
        axes[i,1].imshow(mis, vmin=0, vmax=1)
        
    plt.savefig('depth_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()
#     print('oeverlay imgs lidar shape' , lidar.shape)

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
#     print('blended_img shape' , np.asarray(blended_img).shape)
    # io.imshow(blended_img)
    # io.show()
    # plt.figure()
    # plt.imshow(lidar_color)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img , rgb , lidar_color

def points2depthmap_grid(points, height, width):
    downsample =1
    grid_config = {
                    'x': [-51.2, 51.2, 0.8],
                    'y': [-51.2, 51.2, 0.8],
                    'z': [-5, 3, 8],
                    'depth': [1.0, 60.0, 0.5], # original
                    # 'depth': [1.0, 80.0, 0.5],
                }
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), device=points.device, dtype=torch.float32)
    depth_map_grid = torch.zeros((height, width ,2), device=points.device, dtype=torch.float32)
    coor = torch.round(points[:, :2] / downsample) # uv
    depth = points[:, 2] # z
    # kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
    #     coor[:, 1] >= 0) & (coor[:, 1] < height) & (
    #         depth < grid_config['depth'][1]) & (
    #             depth >= grid_config['depth'][0])
    # coor, depth = coor[kept1], depth[kept1]
    # ranks = coor[:, 0] + coor[:, 1] * width
    # sort = (ranks + depth / 100.).argsort()
    # coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    # kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    # kept2[1:] = (ranks[1:] != ranks[:-1])
    # coor_float, depth = coor[kept2], depth[kept2]
    # coor_long = coor.to(torch.long)
    # depth_map[coor_long[:, 1], coor_long[:, 0]] = depth
    # depth_map_grid[coor_long[:, 1], coor_long[:, 0], :] = coor
    return coor, depth

import numpy as np

def points2depthmap_cpu(points, height, width, downsample=1):
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 0.5],
    }
    height, width = height // downsample, width // downsample
    depth_map = np.zeros((height, width), dtype=np.float32)
    coor = np.round(points[:, :2] / downsample)
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height) & (
            depth < grid_config['depth'][1]) & (
                depth >= grid_config['depth'][0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = np.argsort(ranks + depth / 100.)
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = np.ones(coor.shape[0], dtype=bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    
    coor = coor.astype(np.int64)
    depth_map[coor[:, 1], coor[:, 0]] = depth

    # 최종 유효 인덱스 계산
    valid_indices = np.where(kept1)[0][kept2]
    
    return depth_map, coor, depth, valid_indices

def points2depthmap_gpu(points, height, width, downsample=1):
    device = points.device
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 0.5],
    }
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    
    # 모든 연산을 GPU에서 수행하도록 명시적으로 device 지정
    coor = torch.round(points[:, :2] / downsample).to(device)
    depth = points[:, 2].to(device)
    
    # 벡터화된 연산으로 kept1 계산
    kept1 = ((coor[:, 0] >= 0) & (coor[:, 0] < width) &
             (coor[:, 1] >= 0) & (coor[:, 1] < height) &
             (depth < grid_config['depth'][1]) &
             (depth >= grid_config['depth'][0]))
    
    coor, depth = coor[kept1], depth[kept1]
    
    # width를 텐서로 변환하여 GPU에서 연산
    width_tensor = torch.tensor(width, device=device)
    ranks = coor[:, 0] + coor[:, 1] * width_tensor
    
    # argsort 연산은 이미 GPU에서 수행됨
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    # 벡터화된 연산으로 kept2 계산
    kept2 = torch.ones(coor.shape[0], device=device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    
    coor, depth = coor[kept2], depth[kept2]
    
    coor = coor.to(torch.long)
    
    # GPU에서 인덱싱 연산 수행
    depth_map[coor[:, 1], coor[:, 0]] = depth

    # # 최종 유효 인덱스 계산
    valid_indices = torch.where(kept1)[0][kept2]
    
    return depth_map ,coor, depth, valid_indices

def add_calibration(lidar2img, points_lidar): 
    points_img = points_lidar.tensor[:, :3].matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)

    return points_img


def add_calibration_cpu(lidar2img, points_lidar): 
    # NumPy 행렬 곱셈 사용
    points_img = np.dot(points_lidar[:, :3], lidar2img[:3, :3].T) + lidar2img[:3, 3]
    
    # 투영 좌표 계산
    points_img = np.hstack([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]])

    return points_img


# def add_mis_calibration(extrinsic, intrinsic, points_lidar):
#     max_r = 0.
#     max_t = 0.
#     # 회전 각도 랜덤 생성 (deg 단위)
#     max_angle = max_r  # 최대 회전 각도
#     rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
#     roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
#     rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    
#     # 회전 행렬 생성
#     Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
#                 [np.sin(rotz), np.cos(rotz), 0],
#                 [0, 0, 1]])
#     Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
#                 [0, 1, 0],
#                 [-np.sin(roty), 0, np.cos(roty)]])
#     Rx = np.array([[1, 0, 0],
#                 [0, np.cos(rotx), -np.sin(rotx)],
#                 [0, np.sin(rotx), np.cos(rotx)]])
    
#     # 총 회전 행렬 생성
#     R = np.dot(Rz, np.dot(Ry, Rx))
    
#     # 이동 벡터 랜덤 생성
#     transl_x = np.random.uniform(-max_t, max_t)
#     transl_y = np.random.uniform(-max_t, max_t)
#     transl_z = np.random.uniform(-max_t, max_t)
#     T = np.array([transl_x, transl_y, transl_z])

#     # lidar2img 행렬에 mis-calibration 적용
#     lidar2img_mis_calibrated = extrinsic.clone()
#     lidar2img_mis_calibrated[:3, :3] = torch.tensor(R, dtype=torch.float32) @ extrinsic[:3, :3]
#     lidar2img_mis_calibrated[:3, 3] += torch.tensor(T, dtype=torch.float32)

#     # intrinsic 변환
#     homo_intrinsic = torch.eye(4,dtype=torch.float32)
#     # homo_intrinsic[:3,:3] = intrinsic
#     homo_intrinsic = intrinsic
#     KT = homo_intrinsic.matmul(lidar2img_mis_calibrated)

#     # Mis-calibrated depth map 계산
#     points_img_mis_calibrated = points_lidar.tensor[:, :3].matmul(KT[:3, :3].T) + KT[:3, 3].unsqueeze(0)
#     points_img_mis_calibrated = torch.cat([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]], 1)
#     # points_img_mis_calibrated = points_img_mis_calibrated.matmul(post_rots[cid].T) + post_trans[cid:cid + 1, :]
#     # mis_calibrated_depth_map, uv , z = points2depthmap(points_img_mis_calibrated, imgs.shape[2], imgs.shape[3])
    
#     return points_img_mis_calibrated , KT 

def add_mis_calibration(extrinsic, intrinsic, points_lidar, max_r=1.0, max_t=0.1):
    """
    Apply mis-calibration to lidar points.
    
    Args:
    extrinsic (torch.Tensor): Shape (4, 4), extrinsic matrix
    intrinsic (torch.Tensor): Shape (3, 3) or (4, 4), intrinsic matrix
    points_lidar (torch.Tensor): Shape (N, 3) or (N, 4), lidar points
    max_r (float): Maximum rotation angle in degrees
    max_t (float): Maximum translation distance
    
    Returns:
    points_img_mis_calibrated (torch.Tensor): Mis-calibrated points in image space
    KT (torch.Tensor): Mis-calibrated projection matrix
    """
    device = extrinsic.device
    dtype = extrinsic.dtype
    
    # Generate random rotation angles
    angles = torch.rand(3, device=device, dtype=dtype) * 2 * max_r - max_r
    angles = angles * (torch.pi / 180.0)  # Convert to radians
    
    # Create rotation matrices
    Rx = torch.eye(3, device=device, dtype=dtype)
    Ry = torch.eye(3, device=device, dtype=dtype)
    Rz = torch.eye(3, device=device, dtype=dtype)
    
    if max_r > 0:
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                           [0, torch.sin(angles[0]), torch.cos(angles[0])]], device=device, dtype=dtype)
        
        Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                           [0, 1, 0],
                           [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], device=device, dtype=dtype)
        
        Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                           [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                           [0, 0, 1]], device=device, dtype=dtype)
    
    R = Rz @ Ry @ Rx
    
    # Generate random translation
    T = torch.zeros(3, device=device, dtype=dtype)
    if max_t > 0:
        T = (torch.rand(3, device=device, dtype=dtype) * 2 * max_t - max_t)
    
    # Create homogeneous transformation matrix (4x4) for R and T
    RT = torch.eye(4, device=device, dtype=dtype)
    RT[:3, :3] = R
    RT[:3, 3] = T

    # Apply mis-calibration to extrinsic matrix
    lidar2img_mis_calibrated = extrinsic.clone()
    lidar2img_mis_calibrated[:3, :3] = R @ extrinsic[:3, :3]
    lidar2img_mis_calibrated[:3, 3] += T
    
    # Ensure intrinsic is 4x4
    if intrinsic.shape == (3, 3):
        homo_intrinsic = torch.eye(4, device=device, dtype=dtype)
        homo_intrinsic[:3, :3] = intrinsic
    else:
        homo_intrinsic = intrinsic
    
    # KT = homo_intrinsic @ lidar2img_mis_calibrated
    KT = homo_intrinsic @ lidar2img_mis_calibrated.T
    ##### ref : K(T.t): [results['intrinsics'][i] @ results['extrinsics'][i].T 
    
    # Project points
    if points_lidar.shape[1] == 3:
        points_lidar = torch.cat([points_lidar, torch.ones(points_lidar.shape[0], 1, device=device, dtype=dtype)], dim=1)
    
    # points_img_mis_calibrated = (KT @ points_lidar.tensor.T).T
    # points_img_mis_calibrated = points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3]
    points_img_mis_calibrated = points_lidar.tensor[:, :3].matmul(KT[:3, :3].T) + KT[:3, 3].unsqueeze(0)
    points_img_mis_calibrated = torch.cat([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]], 1)
    
    return points_img_mis_calibrated,RT

def add_mis_calibration_cpu(extrinsic, intrinsic, points_lidar, max_r=1.0, max_t=0.1):
    """
    Apply mis-calibration to lidar points.
    
    Args:
    extrinsic (np.ndarray): Shape (4, 4), extrinsic matrix
    intrinsic (np.ndarray): Shape (3, 3) or (4, 4), intrinsic matrix
    points_lidar (np.ndarray): Shape (N, 3) or (N, 4), lidar points
    max_r (float): Maximum rotation angle in degrees
    max_t (float): Maximum translation distance
    
    Returns:
    points_img_mis_calibrated (np.ndarray): Mis-calibrated points in image space
    RT (np.ndarray): Mis-calibrated transformation matrix
    """
    # Generate random rotation angles
    angles = np.random.rand(3) * 2 * max_r - max_r
    angles = angles * (np.pi / 180.0)  # Convert to radians
    
    # Create rotation matrices
    Rx = np.eye(3)
    Ry = np.eye(3)
    Rz = np.eye(3)
    
    if max_r > 0:
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # Generate random translation
    T = np.zeros(3)
    if max_t > 0:
        T = (np.random.rand(3) * 2 * max_t - max_t)
    
    # Create homogeneous transformation matrix (4x4) for R and T
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = T

    # Apply mis-calibration to extrinsic matrix
    lidar2img_mis_calibrated = extrinsic.copy()
    lidar2img_mis_calibrated[:3, :3] = R @ extrinsic[:3, :3]
    lidar2img_mis_calibrated[:3, 3] += T
    
    # Ensure intrinsic is 4x4
    if intrinsic.shape == (3, 3):
        homo_intrinsic = np.eye(4)
        homo_intrinsic[:3, :3] = intrinsic
    else:
        homo_intrinsic = intrinsic
    
    KT = homo_intrinsic @ lidar2img_mis_calibrated.T
    
    # Project points
    if points_lidar.shape[1] == 3:
        points_lidar = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
    
    points_img_mis_calibrated = np.dot(points_lidar[:, :3], KT[:3, :3].T) + KT[:3, 3]
    points_img_mis_calibrated = np.hstack([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]])
    
    return points_img_mis_calibrated, RT


def apply_random_rt_to_point_cloud(point_cloud, max_r=7.5, max_t=0.20):
    # 회전 각도 랜덤 생성 (deg 단위)
    max_angle = max_r  # 최대 회전 각도
    rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)

    # 회전 행렬 생성
    Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                   [np.sin(rotz), np.cos(rotz), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
                   [0, 1, 0],
                   [-np.sin(roty), 0, np.cos(roty)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotx), -np.sin(rotx)],
                   [0, np.sin(rotx), np.cos(rotx)]])

    # 총 회전 행렬 생성
    R = np.dot(Rz, np.dot(Ry, Rx))

    # 이동 벡터 랜덤 생성
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    T = np.array([transl_x, transl_y, transl_z])
    
    # 포인트 클라우드에 회전 및 이동 변환 적용
    rotated_points = np.dot(point_cloud, R.T)
    transformed_points = rotated_points + T

    return transformed_points

def project_points_to_image(points, camera_intrinsics, camera_extrinsics, image_shape):
    """
    포인트 클라우드를 카메라 이미지에 투영하는 함수
    points: (N, 3) 크기의 포인트 클라우드 텐서
    camera_intrinsics: (3, 3) 크기의 카메라 내부 행렬
    camera_extrinsics: (4, 4) 크기의 카메라 외부 행렬
    image_shape: (height, width) 이미지 크기
    """
    # # 포인트 클라우드에 동차 좌표 추가
    # ones = torch.ones((points.shape[0], 1), dtype=torch.float32)
    # points_homogeneous = torch.cat([points, ones], dim=1)  # (N, 4)

    # # 카메라 좌표계로 변환
    # points_camera = (camera_extrinsics @ points_homogeneous.T).T[:, :3]

    # 이미지 평면으로 투영
    points_image = (camera_intrinsics @ points.T).T

    # 동차 좌표에서 유클리드 좌표로 변환
    uv = points_image[:, :2] / points_image[:, 2:3]
    z = points_image[:, 2]  # z 값 추출

    # 이미지 범위 내의 포인트만 필터링
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < image_shape[1]) & \
           (uv[:, 1] >= 0) & (uv[:, 1] < image_shape[0])
    
    points_image = points_image[mask]
    uv = uv[mask]
    z  = z[mask]
    
    return points_image ,uv ,z

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

def dense_map_cpu_optimized(Pts, n, m, grid):
    ng = 2 * grid + 1  # grid=4 → ng=9
    epsilon = 1e-8
    
    # 유효 영역 재정의 (핵심 수정)
    h = m - ng  # 512 - 9 = 503
    w = n - ng  # 1408 - 9 = 1399

    # 초기화
    mX = np.full((m, n), np.inf, dtype=np.float32)
    mY = np.full((m, n), np.inf, dtype=np.float32)
    mD = np.zeros((m, n), dtype=np.float32)
    
    # 인덱스 클리핑 (중요!)
    mX_idx = np.clip(Pts[0].astype(int), 0, m-1)
    mY_idx = np.clip(Pts[1].astype(int), 0, n-1)
    mX[mX_idx, mY_idx] = Pts[0] - np.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - np.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    # 텐서 구조 조정 (차원 일치 보장)
    KmX = np.zeros((ng, ng, h, w), dtype=np.float32)
    KmY = np.zeros_like(KmX)
    KmD = np.zeros_like(KmX)

    # 슬라이싱 방식 개선 (오류 근본 해결)
    for i in range(ng):
        for j in range(ng):
            row_slice = slice(i, i + h)  # 0 ≤ i ≤ 8 → 0~503, 1~504, ..., 8~511
            col_slice = slice(j, j + w)  # 0 ≤ j ≤ 8 → 0~1399, 1~1400, ..., 8~1407
            KmX[i,j] = mX[row_slice, col_slice] - (grid - i)
            KmY[i,j] = mY[row_slice, col_slice] - (grid - j)
            KmD[i,j] = mD[row_slice, col_slice]

    # 역제곱 가중 평균
    S = np.sum(1 / (KmX**2 + KmY**2 + epsilon), axis=(0,1))
    Y = np.sum(KmD / (KmX**2 + KmY**2 + epsilon), axis=(0,1))
    
    out = np.zeros((m, n), dtype=np.float32)
    out[ng//2 : ng//2 + h, ng//2 : ng//2 + w] = Y / np.where(S==0, 1, S)
    
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
    # colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32)
    colormapped_tensor = torch.from_numpy(colormapped_im)
    return colormapped_tensor

def colormap_cpu(disp):
    """Color mapping for disp -- [H, W] -> [3, H, W]"""
    disp_np = disp  # 이미 NumPy 배열이라고 가정
    vmin = disp_np.min()
    vmax = disp_np.max()
    normalizer = plt.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  # magma, plasma, etc.
    colormapped_im = mapper.to_rgba(disp_np)[:, :, :3]
    return colormapped_im

def trim_corrs(points ,num_kp=30000):
    length = points.shape[0]
#         print ("number of keypoint before trim : {}".format(length))
    if length >= num_kp:
        # mask = np.random.choice(length, num_kp)
        mask = torch.randperm(length)[:num_kp]
        return points[mask]
    else:
        # mask = np.random.choice(length, num_kp - length)
        mask = torch.randint(0, length, (num_kp - length,))
        # return np.concatenate([points, points[mask]], axis=0)
        return torch.cat([points, points[mask]], dim=0)

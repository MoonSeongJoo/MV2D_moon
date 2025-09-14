import torch
import os
from collections import OrderedDict

# --- 설정할 변수들 ---

# 1. 베이스가 될 전체 모델 체크포인트
base_ckpt_path = 'data/work_dirs/20240811_mv2d_modi_cameraonly_base/latest.pth'

# 2. 새로 끼워넣을 부품(모듈) 체크포인트 정보
component_ckpts = [
    {
        'prefix': 'roi_head.lidar_voxelnet.',
        'ckpt_path': 'data/weights/lidar_backbone_neck_rev1.0.pth'
    },
    {
        'prefix': 'roi_head.bbox_head.transformer_lidar.',
        # ‼️ 경로를 '_corrected.pth' 버전으로 사용하는 것을 권장합니다.
        'ckpt_path': 'data/weights/transformer_lidar_corrected.pth' 
    }
]

# 3. 최종적으로 저장할 통합 체크포인트 파일 경로
save_path = 'data/weights/merged_checkpoint.pth'

# --- 스크립트 실행 부분 ---

print(f"🔩 1. 베이스 체크포인트 로딩: {base_ckpt_path}")
base_checkpoint = torch.load(base_ckpt_path, map_location='cpu')
base_state_dict = base_checkpoint['state_dict']

# 부품별로 가중치를 순회하며 베이스에 업데이트
for component in component_ckpts:
    prefix = component['prefix']
    ckpt_path = component['ckpt_path']
    print(f"\n⚙️ 2. 부품 체크포인트 처리: {ckpt_path}")
    print(f"   - '{prefix}' 경로에 가중치를 주입합니다.")
    
    comp_checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # --- ‼️ 수정된 부분 1: 'state_dict' 키 존재 여부 확인 ---
    # 'state_dict' 키가 있으면 그 값을 사용하고, 없으면 전체를 state_dict로 간주
    if 'state_dict' in comp_checkpoint:
        comp_state_dict = comp_checkpoint['state_dict']
    else:
        comp_state_dict = comp_checkpoint
    
    updated_keys = 0
    for key, value in comp_state_dict.items():
        new_key = prefix + key
        
        # --- ‼️ 수정된 부분 2: 키 존재 여부와 상관없이 무조건 추가/업데이트 ---
        # if new_key in base_state_dict: # <-- 이 조건문을 제거
        base_state_dict[new_key] = value
        updated_keys += 1
            
    print(f"   - ✅ {updated_keys}개의 파라미터를 성공적으로 주입(추가 또는 업데이트)했습니다.")

# 원본 체크포인트의 state_dict를 업데이트된 버전으로 교체
base_checkpoint['state_dict'] = base_state_dict

print(f"\n💾 3. 병합된 체크포인트 저장: {save_path}")
torch.save(base_checkpoint, save_path)
print("🎉 모든 작업이 완료되었습니다!")
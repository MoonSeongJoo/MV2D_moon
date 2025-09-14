import torch
from collections import OrderedDict

# 1. 기존 체크포인트(가중치 파일) 로드
checkpoint_path = 'data/weights/lidar_backbone_rev1.0.pth'  # 원본 가중치 파일 경로
corrected_checkpoint_path = 'data/weights/lidar_backbone_rev2.0.pth' # 새로 저장할 파일 경로
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 원본 state_dict 추출
# print(checkpoint.keys()) 로 확인 후 맞게 수정하세요.
original_state_dict = checkpoint['state_dict'] 

# 2. 새로운 state_dict 생성 및 키 값 변경
new_state_dict = OrderedDict()
prefix_to_remove = 'roi_head.lidar_voxelnet.'

for k, v in original_state_dict.items():
    if k.startswith(prefix_to_remove):
        # 접두사를 제거한 새로운 키를 만듭니다.
        new_key = k.replace(prefix_to_remove, '', 1)
        new_state_dict[new_key] = v
    else:
        # 접두사가 없는 다른 키가 있을 경우를 대비해 그대로 추가합니다.
        new_state_dict[k] = v

# 3. 수정된 state_dict를 새로운 파일로 저장
# 원본 파일의 다른 정보(epoch 등)도 유지하고 싶다면 아래와 같이 업데이트합니다.
# checkpoint['state_dict'] = new_state_dict
# torch.save(checkpoint, corrected_checkpoint_path)

# state_dict만 따로 저장하고 싶다면 아래 코드를 사용합니다.
torch.save({'state_dict': new_state_dict}, corrected_checkpoint_path)

print(f"키 값 변경 후 새로운 가중치 파일을 '{corrected_checkpoint_path}'에 성공적으로 저장했습니다.")

# 이제부터는 아래와 같이 corrected_checkpoint_path 파일을 바로 로드해서 사용하면 됩니다.
# model = YourModelClass()
# checkpoint_corrected = torch.load(corrected_checkpoint_path)
# model.load_state_dict(checkpoint_corrected['state_dict'])
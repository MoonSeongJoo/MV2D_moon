import torch
import os
from collections import OrderedDict

def convert_state_dict_keys(state_dict):
    """
    state_dict에서 'pts_backbone.' 접두사를 'backbone_3d.'로 교체합니다.
    """
    new_state_dict = OrderedDict()
    prefix_to_remove = 'pts_backbone.'
    
    # 모델이 실제로 필요로 하는 키 접두사로 수정합니다.
    new_prefix = 'backbone_3d.' 
    
    print("🚀 키 변환 시작 (접두사 교체 방식)...")

    for key, value in state_dict.items():
        # 1. 키가 'pts_backbone.'으로 시작하는지 확인
        if key.startswith(prefix_to_remove):
            # 2. 기존 접두사를 제거하고 새로운 접두사를 앞에 붙여 새 키를 생성
            base_key = key[len(prefix_to_remove):]
            new_key = new_prefix + base_key
            new_state_dict[new_key] = value
            # print(f"  ✅ {key} -> {new_key}") # 변환 로그 확인 시 주석 해제

    print(f"✅ 키 변환 완료! 총 {len(new_state_dict)}개의 백본 파라미터가 변환되었습니다.")
    return new_state_dict

def main():
    checkpoint_path = "/workspace/MV2D_moon/data/weights/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
    # 혼동을 피하기 위해 저장 파일 이름을 변경하는 것을 추천합니다.
    save_path = "/workspace/MV2D_moon/data/weights/converted_backbone_weights_final.pth"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint 파일 없음: {checkpoint_path}")
        return

    print(f"📂 Checkpoint 로드 중: {checkpoint_path}")
    original_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in original_checkpoint:
        state_dict = original_checkpoint['state_dict']
    else:
        state_dict = original_checkpoint

    # 키 변환 함수 호출
    converted_state_dict = convert_state_dict_keys(state_dict)

    # 변환된 가중치 저장
    if converted_state_dict:
        torch.save(converted_state_dict, save_path)
        print(f"💾 변환된 가중치 저장 완료: {save_path}")
        print("\n🎉 변환 및 저장 성공!")
    else:
        print("\n❌ 변환할 백본 키를 찾지 못했습니다. Checkpoint 파일의 키를 확인해주세요.")


if __name__ == "__main__":
    main()
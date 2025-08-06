import torch
import os
import re

def convert_key(key):
    # backbone_3d.conv_input.x.y  →  blocks.0.{x*3}.{param}
    # m = re.match(r'backbone_3d\\.conv_input\\.(\\d+)\\.(\\w+)', key)
    pattern = r'backbone_3d\.conv_input\.(\d+)\.(\w+)'  # r-string으로 작성
    m = re.match(pattern, key)
    if m:
        idx, param = m.groups()
        return f'blocks.0.{int(idx)*3}.{param}'
    # backbone_3d.convN.x.y.param  →  blocks.{N}.{x*3+y}.{param}
    m = re.match(r'backbone_3d\\.conv([0-9]+)\\.(\\d+)\\.(\\d+)\\.(\\w+)', key)
    if m:
        stage, block, subblock, param = m.groups()
        block_idx = int(stage)
        sub_idx = int(block)*3 + int(subblock)
        return f'blocks.{block_idx}.{sub_idx}.{param}'
    # backbone_3d.convN.x.y.num_batches_tracked  →  blocks.{N}.{x*3+y}.num_batches_tracked
    m = re.match(r'backbone_3d\\.conv([0-9]+)\\.(\\d+)\\.(\\d+)\\.(num_batches_tracked)', key)
    if m:
        stage, block, subblock, param = m.groups()
        block_idx = int(stage)
        sub_idx = int(block)*3 + int(subblock)
        return f'blocks.{block_idx}.{sub_idx}.{param}'
    return None  # 변환 불가 키는 무시

def extract_and_convert_from_model_state(checkpoint_path, save_path):
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state' not in checkpoint:
        print("❌ 'model_state' key가 없습니다.")
        return False
    state_dict = checkpoint['model_state']
    print(f"✅ 'model_state' key 확인됨, 총 파라미터 개수: {len(state_dict)}")

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone_3d'):
            new_key = convert_key(key)
            if new_key is not None:
                new_state_dict[new_key] = value
                print(f"✅ 변환: {key} -> {new_key}")
        else:
            # backbone_3d로 시작하지 않는 키들은 필요시 별도 처리하거나 스킵
            pass

    torch.save(new_state_dict, save_path)
    print(f"💾 변환된 가중치 저장 완료: {save_path}")
    return True

def main():
    checkpoint_path = "/workspace/MV2D_moon/data/weights/second_iou7909.pth"
    save_path = "/workspace/MV2D_moon/data/weights/converted_second_7862.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint 파일 없음: {checkpoint_path}")
        return
    success = extract_and_convert_from_model_state(checkpoint_path, save_path)
    if success:
        print("\n🎉 변환 및 저장 성공!")
    else:
        print("\n❌ 변환 실패!")

if __name__ == "__main__":
    main()

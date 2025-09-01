import torch
import os
from collections import OrderedDict

def extract_and_rename_weights(checkpoint_path, save_path, prefix_to_remove):
    """
    체크포인트에서 특정 접두사를 가진 가중치만 추출하고,
    해당 접두사를 제거하여 새로운 파일로 저장합니다.
    
    Args:
        checkpoint_path: 원본 체크포인트 파일 경로
        save_path: 저장할 파일 경로
        prefix_to_remove: 필터링 및 제거할 접두사
    """
    
    # 1. 체크포인트 로딩
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 2. State dict 추출
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✅ Found 'state_dict' key")
    else:
        state_dict = checkpoint
        print("✅ Using the entire checkpoint as state_dict")

    # 3. 전체 키 확인
    print(f"📊 Total keys in original state_dict: {len(state_dict)}")
    
    # 4. 가중치 필터링 및 키 이름 변경
    renamed_state_dict = OrderedDict()
    
    print("\n" + "="*60)
    print(f"🚀 Filtering keys with prefix: '{prefix_to_remove}'")
    print("="*60)

    for key, value in state_dict.items():
        if key.startswith(prefix_to_remove):
            # 접두사를 제거하여 새로운 키 생성
            new_key = key.replace(prefix_to_remove, '', 1)
            renamed_state_dict[new_key] = value
            print(f"✅ Extracted & Renamed: {key}  ->  {new_key}")
    
    # 5. 추출 결과 요약
    print("\n" + "="*60)
    print("📋 EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total extracted parameters: {len(renamed_state_dict)}")
    print(f"Original total parameters: {len(state_dict)}")
    if len(state_dict) > 0 and len(renamed_state_dict) > 0:
        print(f"Extraction ratio: {len(renamed_state_dict)/len(state_dict)*100:.2f}%")
    
    # 6. 추출된 가중치가 있는지 확인
    if not renamed_state_dict:
        print("\n⚠️ WARNING: No matching keys found to extract!")
        print("Available keys in checkpoint (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {i+1}. {key}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict)-10} more keys")
        return False
    
    # 7. 새로운 파일로 저장
    torch.save({'state_dict': renamed_state_dict}, save_path)
    print(f"\n💾 Successfully saved renamed weights to: {save_path}")
    
    # 8. 저장된 파일 크기 확인
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"📁 Saved file size: {file_size:.2f} MB")
    
    return True

def verify_extracted_weights(saved_path):
    """
    저장된 가중치 파일 검증
    """
    print("\n" + "="*60)
    print("🔍 VERIFICATION")
    print("="*60)
    
    try:
        checkpoint = torch.load(saved_path, map_location='cpu')
        loaded_weights = checkpoint['state_dict']
        
        print(f"✅ Successfully loaded: {saved_path}")
        print(f"📊 Number of parameters: {len(loaded_weights)}")
        
        print("\n📋 Renamed keys (first 10):")
        for i, key in enumerate(list(loaded_weights.keys())[:10]):
            print(f"  {i+1}. {key}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

# 사용 예시
def main():
    """
    메인 실행 함수
    """
    # 1. 원본 체크포인트 파일 경로
    checkpoint_path = "data/work_dirs/20250825_lidar_only/latest.pth"
    
    # 2. 새로 저장할 파일 경로
    save_path = "data/weights/transformer_lidar_corrected.pth"
    
    # 3. 제거할 접두사 지정 <-- ‼️ 이 부분을 수정했습니다.
    prefix_to_remove = 'roi_head.bbox_head.transformer_lidar.'
    
    # 원본 파일 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    # 가중치 추출 및 이름 변경 후 저장
    success = extract_and_rename_weights(checkpoint_path, save_path, prefix_to_remove)
    
    if success:
        # 저장된 파일 검증
        verify_extracted_weights(save_path)
        print("\n🎉 Weight extraction and renaming completed successfully!")
    else:
        print("\n❌ Weight extraction and renaming failed!")

if __name__ == "__main__":
    main()
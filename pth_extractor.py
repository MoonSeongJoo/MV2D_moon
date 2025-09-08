import torch
import os

def extract_base_detector_and_neck(checkpoint_path, save_path):
    """
    epoch_19.pth에서 base_detector와 neck 가중치만 추출하여 저장
    
    Args:
        checkpoint_path: 원본 체크포인트 파일 경로 (epoch_19.pth)
        save_path: 저장할 파일 경로 (a.pth)
    """
    
    # 1. 체크포인트 로딩
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 2. State dict 추출
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✅ Found 'state_dict' key")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("✅ Found 'model' key")
    else:
        state_dict = checkpoint
        print("✅ Using checkpoint as state_dict")
    
    # 3. 전체 키 확인
    print(f"📊 Total keys in checkpoint: {len(state_dict)}")
    
    # 4. base_detector와 neck 키만 필터링
    filtered_state_dict = {}
    
    base_detector_count = 0
    neck_count = 0
    
    for key, value in state_dict.items():
        if key.startswith('roi_head.z_estimator'):
            filtered_state_dict[key] = value
            
            # if key.startswith('base_detector'):
            #     base_detector_count += 1
            # elif key.startswith('neck'):
            #     neck_count += 1
            if key.startswith('z_estimator'):
                corr_count += 1
            
            print(f"✅ Extracted: {key}")
    
    # 5. 추출 결과 요약
    print("\n" + "="*60)
    print("📋 EXTRACTION SUMMARY")
    print("="*60)
    print(f"Base Detector parameters: {base_detector_count}")
    print(f"Neck parameters: {neck_count}")
    print(f"Total extracted parameters: {len(filtered_state_dict)}")
    print(f"Original total parameters: {len(state_dict)}")
    print(f"Extraction ratio: {len(filtered_state_dict)/len(state_dict)*100:.2f}%")
    
    # 6. 추출된 가중치가 있는지 확인
    if len(filtered_state_dict) == 0:
        print("⚠️ WARNING: No matching keys found!")
        print("Available keys in checkpoint:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {i+1}. {key}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict)-10} more keys")
        return False
    
    # 7. 새로운 파일로 저장
    torch.save(filtered_state_dict, save_path)
    print(f"💾 Successfully saved filtered weights to: {save_path}")
    
    # 8. 저장된 파일 크기 확인
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"📁 Saved file size: {file_size:.2f} MB")
    
    return True

def verify_extracted_weights(saved_path):
    """
    저장된 가중치 파일 검증
    
    Args:
        saved_path: 저장된 파일 경로
    """
    print("\n" + "="*60)
    print("🔍 VERIFICATION")
    print("="*60)
    
    try:
        # 저장된 파일 로딩
        loaded_weights = torch.load(saved_path, map_location='cpu')
        
        print(f"✅ Successfully loaded: {saved_path}")
        print(f"📊 Number of parameters: {len(loaded_weights)}")
        
        # 키 목록 출력
        print("\n📋 Available keys:")
        for i, key in enumerate(loaded_weights.keys()):
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
    checkpoint_path = "data/work_dirs/20240811_mv2d_modi_cameraonly_base/epoch_72.pth"
    save_path = "data/weights/zestimator.pth"
    
    # 파일 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    # 가중치 추출 및 저장
    success = extract_base_detector_and_neck(checkpoint_path, save_path)
    
    if success:
        # 저장된 파일 검증
        verify_extracted_weights(save_path)
        print("\n🎉 Weight extraction completed successfully!")
    else:
        print("\n❌ Weight extraction failed!")

if __name__ == "__main__":
    main()

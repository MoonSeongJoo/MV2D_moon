import mmcv
import matplotlib.pyplot as plt
from PIL import Image

filenames = './data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-18-11-50-34+0800__CAM_FRONT_RIGHT__1531886334370339.jpg'
valid_images = []
valid_indices = []

try :
    img = mmcv.imread('./data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-18-11-50-34+0800__CAM_FRONT_RIGHT__1531886334370339.jpg')
    if img is None:
        raise ValueError(f"Image is corrupted or cannot be read.")
    valid_images.append(img)
    # valid_indices.append(idx)
except Exception as e:
    print(f"Skipping file due to error: {e}")

img = mmcv.imread('./data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-18-11-50-34+0800__CAM_FRONT_RIGHT__1531886334370339.jpg')


img2 = mmcv.imread('./data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-08-01-15-10-21+0800__CAM_FRONT_RIGHT__1533107699020339.jpg')
img1 = Image.open('./data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-18-11-50-34+0800__CAM_FRONT_RIGHT__1531886334370339.jpg')

# matplotlib을 사용하여 이미지 표시
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')  # 축 레이블 제거
plt.title('NuScenes Front Right Camera Image')

# 이미지를 PNG 형식으로 저장
plt.savefig('1.png', format='png', dpi=300, bbox_inches='tight')

print(img)
print("end")
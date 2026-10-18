import cv2
import numpy as np
from PIL import Image, ImageEnhance

# 读取图片
input_path = "/data1/liuyeqiang/avis-main/5.png"
img = cv2.imread(input_path, cv2.IMREAD_COLOR)

# 先放大 6 倍 (Lanczos 插值)
scale_factor = 6
new_size = (img.shape[1] * scale_factor, img.shape[0] * scale_factor)
upscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

# 高提升锐化（High-boost filtering）
# 1. 高斯模糊
gauss = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0, sigmaY=1.0)
# 2. 创建高提升图像
alpha = 1.5  # 锐化强度
high_boost = cv2.addWeighted(upscaled, 1 + alpha, gauss, -alpha, 0)

# 额外细节增强（Unsharp Mask）
sharp_kernel = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(high_boost, -1, sharp_kernel)

# 轻微对比度提升
pil_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
contrast_enhancer = ImageEnhance.Contrast(pil_img)
final_img = contrast_enhancer.enhance(1.1)

# 保存
output_path = "/data1/liuyeqiang/avis-main/5_upscaled_sharpened.png"
final_img.save(output_path)

output_path

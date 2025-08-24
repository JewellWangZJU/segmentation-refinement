import cv2
import numpy as np
import segmentation_refinement as refine

# 读取原图和mask
image = cv2.imread('datasets/0.png')
mask = cv2.imread('datasets/0_mask.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found! 请检查路径")
if mask is None:
    raise FileNotFoundError("Mask not found! 请检查路径")

# 把原始mask二值化
mask_binary = (mask == 1).astype(np.uint8) * 255

# 初始化Refiner
refiner = refine.Refiner(device='cuda:0')

# 执行精细化
output = refiner.refine(image, mask_binary, fast=False, L=496)

print(np.unique(output))
# =========== ✅ 新增部分: 生成叠加图（修改前） ===========

# 创建红色蒙版
red_overlay = np.zeros_like(image)
red_overlay[:, :, 2] = 255  # R通道

# 提取mask区域
mask_color = cv2.bitwise_and(red_overlay, red_overlay, mask=mask_binary)

# 融合
blended_before = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)

# 保存
cv2.imwrite('output_before.png', blended_before)

# =========== ✅ 新增部分: 生成叠加图（修改后） ===========
# refine输出是BGR，要先转灰度
output_gray = output
output_binary = (output_gray > 127).astype(np.uint8) * 255

# 创建红色蒙版
mask_color_after = cv2.bitwise_and(red_overlay, red_overlay, mask=output_binary)

# 融合
blended_after = cv2.addWeighted(image, 0.7, mask_color_after, 0.3, 0)

# 保存
cv2.imwrite('output_after.png', blended_after)

# =========== ✅ 新增部分: 保存refined mask本身 ===========
cv2.imwrite('output_refined_mask.png', output)

# 原来的输出也保留（如果需要）
cv2.imwrite('output.png', output)

print("已生成 output_before.png、output_after.png、output_refined_mask.png")

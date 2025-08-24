import cv2
import matplotlib.pyplot as plt
import segmentation_refinement as refine

# 读取输入图片
image = cv2.imread('datasets/0.png')
mask = cv2.imread('datasets/0_mask.png', cv2.IMREAD_GRAYSCALE)

# 初始化模型
refiner = refine.Refiner(device='cuda:0')  # device可以是'cpu'或'cuda:0'

# 执行精细化
output = refiner.refine(image, mask, fast=False, L=900)

# 注意: output是numpy数组(BGR)，需要转换为RGB再保存
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# 保存输出图片
cv2.imwrite('output.png', output)

print("已将分割结果保存为 output.png")


import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# 大气湍流退化模型
def atmospheric_turbulence_psf(shape, k):
    M, N = shape
    H = np.zeros((M, N), dtype=np.float32)
    u_center = M // 2  # 中心对称坐标
    v_center = N // 2

    for u in range(M):
        for v in range(N):
            distance = ((u - u_center)**2 + (v - v_center)**2) ** (5 / 6)  # 计算频域距离的5/6次方
            H[u, v] = np.exp(-k * distance)
    return H

# 退化图像
def degrade_image(image, H):
    image_fft = fft2(image)          # 对原始图像做傅里叶变换
    degraded_fft = image_fft * H     # 与退化模型相乘
    degraded_image = ifft2(degraded_fft).real  # 反变换回空域
    return degraded_image

# 维纳滤波复原图像
def wiener_filter(degraded_image, H, K=0.01):
    degraded_fft = fft2(degraded_image)
    H_conj = np.conj(H)
    H_abs2 = np.abs(H)**2
    restored_fft = (H_conj / (H_abs2 + K)) * degraded_fft  # 维纳滤波公式
    restored_image = ifft2(restored_fft).real
    return restored_image

# 加载原始图像
image_path = 'C:/Users/Lenovo/Desktop/lena.jpg'  # 替换为你要处理的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
M, N = image.shape

# 生成大气湍流退化模型
k = 0.0025  # 湍流参数
H = atmospheric_turbulence_psf((M, N), k)

# 图像退化处理
degraded_image = degrade_image(image, H)

# 图像复原处理（维纳滤波）
K_value = 0.01  # 噪声功率比参数
restored_image = wiener_filter(degraded_image, H, K=K_value)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Degraded Image")
plt.imshow(np.clip(degraded_image, 0, 255), cmap='gray')  # 限定显示范围
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Restored Image (Wiener Filter)")
plt.imshow(np.clip(restored_image, 0, 255), cmap='gray')  # 限定显示范围
plt.axis('off')

plt.tight_layout()
plt.show()

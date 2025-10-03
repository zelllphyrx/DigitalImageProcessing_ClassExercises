import cv2
import numpy as np
import matplotlib.pyplot as plt

def spectral_residual_saliency(img):
    # 第一步：将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 第二步：对灰度图像进行傅里叶变换
    fft = np.fft.fft2(gray_img)
    magnitude = np.abs(fft)
    phase = np.angle(fft)

    # 第三步：计算对数幅度谱
    log_magnitude = np.log(magnitude + 1e-8)

    # 第四步：对对数幅度谱进行平滑
    smooth_magnitude = cv2.GaussianBlur(log_magnitude, (3, 3), 0)

    # 第五步：计算谱残差
    spectral_residual = log_magnitude - smooth_magnitude

    # 第六步：通过指数函数恢复幅值
    modified_magnitude = np.exp(spectral_residual)

    # 第七步：重新组合频谱并进行逆傅里叶变换
    modified_fft = modified_magnitude * np.exp(1j * phase)
    saliency_map = np.fft.ifft2(modified_fft)

    # 取实部，得到显著性图
    saliency_map = np.abs(saliency_map)

    # 第八步：归一化显著性图
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255
    saliency_map = saliency_map.astype(np.uint8)

    # 第九步：对显著性图进行高斯平滑
    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 0)

    return saliency_map

# 读取图像
img = cv2.imread('C://Users//Lenovo//Desktop//image_1.png')

# 计算显著性图
saliency_map = spectral_residual_saliency(img)

# 显示结果
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(saliency_map, cmap='gray')

plt.show()

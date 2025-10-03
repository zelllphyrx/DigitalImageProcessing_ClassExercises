import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
img = cv2.imread('C:\\Users\\Lenovo\\Desktop\\image_1.png')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为浮点格式并归一化到0到1之间
fgrayimg = np.float32(grayimg) / 255.0

# 进行离散余弦变换 (DCT)
dctimg = cv2.dct(fgrayimg)

# 计算DCT的频谱图，以便进行可视化
# 将频谱值缩放到0到255之间并转换为8位图像
dctimg_visual = cv2.normalize(dctimg, None, 0, 255, cv2.NORM_MINMAX)
dctimg_visual = np.uint8(dctimg_visual)

# 进行逆离散余弦变换 (IDCT)
imgback = cv2.idct(dctimg)

# 显示原始图像、DCT频谱图和重建图像
cv2.imshow('Original Image', grayimg)
cv2.imshow('DCT Spectrum', dctimg_visual)
cv2.imshow('Reconstructed Image', imgback)

# 观察能量分布情况，绘制DCT系数的直方图
plt.figure(figsize=(10, 5))
plt.hist(np.abs(dctimg.flatten()), bins=100, color='blue', alpha=0.7)
plt.title('DCT Coefficients Energy Distribution')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# 等待用户按键以关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

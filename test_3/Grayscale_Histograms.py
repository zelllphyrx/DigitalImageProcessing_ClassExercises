import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
img = cv2.imread('C:\\Users\\Lenovo\\Desktop\\image_1.png', cv2.IMREAD_GRAYSCALE)

# 计算灰度直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 显示灰度直方图
plt.plot(hist, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

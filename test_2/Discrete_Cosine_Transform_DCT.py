import numpy as np
import cv2

# 读取图像
img = cv2.imread('C:\\Users\\Lenovo\\Desktop\\image_1.png')

# 转换图像为灰度图像
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为浮点格式并归一化到0到1之间
fgrayimg = np.float32(grayimg) / 255.0

# 进行离散余弦变换 (DCT)
dctimg = cv2.dct(fgrayimg)
print(dctimg)
# 进行离散余弦变换反变换 (IDCT)
imgback = cv2.idct(dctimg)
print(imgback)
cv2.imshow('original',grayimg) # 显示原始图像
cv2.imshow('dct',dctimg) # 显示DCT频谱图
cv2.imshow('idct',imgback) # 显示IDCT反变换后的图像
# 等待按键关闭窗口
cv2.waitKey(0)
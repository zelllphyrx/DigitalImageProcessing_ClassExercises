import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 灰度熵计算函数
def calculateH(img):
    # 获取图像的行数和列数
    line = img.shape[0]
    row = img.shape[1]
    # 创建一个大小为256的数组，用于统计每个像素值的频数
    s = np.zeros(256)
    # 遍历图像中的每个像素，统计每个灰度值出现的次数
    for i in range(line):
        for j in range(row):
            s[img[i][j]] += 1.0
    # 计算归一化的概率并根据公式计算熵
    for m in range(256):
        if s[m] == 0:  # 如果某个灰度值的频数为0，跳过
            continue
        s[m] = s[m] / (line * row)  # 计算概率
        s[m] = -s[m] * (math.log2(s[m]))  # 计算信息熵
    return sum(s)  # 返回总的熵值

# 以灰度模式读入图片，计算每张图片的熵
img1_1 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-landscape-1.png', 0)  # 读取第一张图片
s11 = calculateH(img1_1)  # 计算该图片的熵
img1_2 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-landscape-2.png', 0)
s12 = calculateH(img1_2)
img1_3 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-landscape-3.png', 0)
s13 = calculateH(img1_3)
img1_4 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-landscape-4.png', 0)
s14 = calculateH(img1_4)
img2_1 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-flower-1.png', 0)
s21 = calculateH(img2_1)
img2_2 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-flower-2.png', 0)
s22 = calculateH(img2_2)
img2_3 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-flower-3.png', 0)
s23 = calculateH(img2_3)
img2_4 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-flower-4.png', 0)
s24 = calculateH(img2_4)
img3_1 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-animal-1.png', 0)
s31 = calculateH(img3_1)
img3_2 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-animal-2.png', 0)
s32 = calculateH(img3_2)
img3_3 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-animal-3.png', 0)
s33 = calculateH(img3_3)
img3_4 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-animal-4.png', 0)
s34 = calculateH(img3_4)

# 显示第一组图片及对应熵值
plt.figure(1)
plt.subplot(221), plt.title(s11), plt.axis('off'), plt.imshow(img1_1, cmap='gray')
plt.subplot(222), plt.title(s12), plt.axis('off'), plt.imshow(img1_2, cmap='gray')
plt.subplot(223), plt.title(s13), plt.axis('off'), plt.imshow(img1_3, cmap='gray')
plt.subplot(224), plt.title(s14), plt.axis('off'), plt.imshow(img1_4, cmap='gray')

# 显示第二组图片及对应熵值
plt.figure(2)
plt.gray()  # 设置全局颜色映射为灰度模式
plt.subplot(221), plt.title(s21), plt.axis('off'), plt.imshow(img2_1, cmap=None)
plt.subplot(222), plt.title(s22), plt.axis('off'), plt.imshow(img2_2, cmap=None)
plt.subplot(223), plt.title(s23), plt.axis('off'), plt.imshow(img2_3, cmap=None)
plt.subplot(224), plt.title(s24), plt.axis('off'), plt.imshow(img2_4, cmap=None)

# 显示第三组图片及对应熵值
plt.figure(3)
plt.gray()  # 设置全局灰度模式
plt.subplot(221), plt.title(s31), plt.axis('off'), plt.imshow(img3_1)
plt.subplot(222), plt.title(s32), plt.axis('off'), plt.imshow(img3_2)
plt.subplot(223), plt.title(s33), plt.axis('off'), plt.imshow(img3_3)
plt.subplot(224), plt.title(s34), plt.axis('off'), plt.imshow(img3_4)
plt.show()  # 显示所有图像

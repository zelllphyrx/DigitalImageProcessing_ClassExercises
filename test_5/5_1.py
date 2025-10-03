import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 自定义的传入灰度图像计算其熵的函数
def calculateH(img):
    line = img.shape[0]
    row = img.shape[1]
    s = np.zeros(256)
    for i in range(line):
        for j in range(row):
            s[img[i][j]] += 1.0
    for m in range(256):
        if s[m] == 0:
            continue
        s[m] = s[m] / (line * row)
        s[m] = -s[m] * (math.log2(s[m]))
    return sum(s)

# 以灰度模式读入图片，计算图像的熵
img1_1 = cv2.imread('C:/Users/Lenovo/Desktop/6-1-landscape-1.png', 0)
s11 = calculateH(img1_1)
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

# 显示1组图片和对应的熵，cmap颜色映射模式为gray （None时是RGB模式）
plt.figure(1)
plt.subplot(221), plt.title(s11), plt.axis('off'), plt.imshow(img1_1, cmap='gray')
plt.subplot(222), plt.title(s12), plt.axis('off'), plt.imshow(img1_2, cmap='gray')
plt.subplot(223), plt.title(s13), plt.axis('off'), plt.imshow(img1_3, cmap='gray')
plt.subplot(224), plt.title(s14), plt.axis('off'), plt.imshow(img1_4, cmap='gray')

# 显示2组图片和对应的熵，思考：为什么显示颜色不正常
plt.figure(2)
plt.gray()
plt.subplot(221), plt.title(s21), plt.axis('off'), plt.imshow(img2_1, cmap=None)
plt.subplot(222), plt.title(s22), plt.axis('off'), plt.imshow(img2_2, cmap=None)
plt.subplot(223), plt.title(s23), plt.axis('off'), plt.imshow(img2_3, cmap=None)
plt.subplot(224), plt.title(s24), plt.axis('off'), plt.imshow(img2_4, cmap=None)

# 显示3组图片和对应的熵
plt.figure(3)
plt.gray()
plt.subplot(221), plt.title(s31), plt.axis('off'), plt.imshow(img3_1)
plt.subplot(222), plt.title(s32), plt.axis('off'), plt.imshow(img3_2)
plt.subplot(223), plt.title(s33), plt.axis('off'), plt.imshow(img3_3)
plt.subplot(224), plt.title(s34), plt.axis('off'), plt.imshow(img3_4)
plt.show()

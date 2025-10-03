import cv2
import matplotlib.pyplot as plt
import pylab

img = cv2.imread('C:/Users/Lenovo/Desktop/6-1-animal-1.png')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(1)
plt.gray()
plt.subplot(111), plt.title("Original"), plt.axis('off'), plt.imshow(grayimg)
# 最高位--第8位
bit8 = grayimg >> 7
# 次高位--第7位
bit7 = (grayimg >> 6) & 1
# 第6位
bit6 = (grayimg >> 5) & 1
# 第5位
bit5 = (grayimg >> 4) & 1
# 第4位
bit4 = (grayimg >> 3) & 1
# 第3位
bit3 = (grayimg >> 2) & 1
# 第2位
bit2 = (grayimg >> 1) & 1
# 第1位
bit1 = grayimg & 1

# 从高位到低位，显示各位平面
plt.figure(2)
plt.subplot(241), plt.title("8Bit Image"), plt.axis('off'), plt.imshow(bit8)
plt.subplot(242), plt.title("7Bit Image"), plt.axis('off'), plt.imshow(bit7)
plt.subplot(243), plt.title("6Bit Image"), plt.axis('off'), plt.imshow(bit6)
plt.subplot(244), plt.title("5Bit Image"), plt.axis('off'), plt.imshow(bit5)
plt.subplot(245), plt.title("4Bit Image"), plt.axis('off'), plt.imshow(bit4)
plt.subplot(246), plt.title("3Bit Image"), plt.axis('off'), plt.imshow(bit3)
plt.subplot(247), plt.title("2Bit Image"), plt.axis('off'), plt.imshow(bit2)
plt.subplot(248), plt.title("1Bit Image"), plt.axis('off'), plt.imshow(bit1)

plt.show()
cv2.waitKey(0)

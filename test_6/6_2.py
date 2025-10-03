import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import math
import cv2

# 仿真运动模糊
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(35):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(35):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()

# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred

def inverse(input, PSF, eps):  # 逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result

def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result

image = cv2.imread('C:/Users/Lenovo/Desktop/lena.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_h = image.shape[0]
img_w = image.shape[1]
graph.figure(1)
graph.xlabel("Original Image")
graph.gray()
graph.imshow(image)  # 显示原图像

graph.figure(2)
graph.gray()
# 进行运动模糊处理
PSF = motion_process((img_h, img_w), 45)
blurred = np.abs(make_blurred(image, PSF, 1e-3))

graph.subplot(231)
graph.xlabel("Motion blurred")
graph.imshow(blurred)

result = inverse(blurred, PSF, 1e-3)  # 逆滤波
graph.subplot(232)
graph.xlabel("inverse deblurred")
graph.imshow(result)

result = wiener(blurred, PSF, 1e-3)  # 维纳滤波
graph.subplot(233)
graph.xlabel("wiener deblurred(k=0.01)")
graph.imshow(result)

# 退化过程二：加入高斯噪声
blurred_noisy = np.random.normal(0, 5, blurred.shape)  # 均值=0，方差=5，高斯噪声

# 添加高斯噪声后去噪
blurred_noisy = blurred + np.random.normal(0, np.sqrt(5), blurred.shape)  # 添加高斯噪声
denoised = cv2.fastNlMeansDenoising(np.uint8(blurred_noisy), None, h=10, templateWindowSize=7, searchWindowSize=21)

# 显示去噪结果
graph.subplot(234)
graph.xlabel("Denoised")
graph.imshow(denoised, cmap='gray')

# 对去噪后的图像进行逆滤波复原
result_inverse = inverse_with_regularization(denoised, PSF, eps=1e-3, regularization=0.01)
graph.subplot(235)
graph.xlabel("Inverse deblurred after denoise")
graph.imshow(result_inverse, cmap='gray')

# 对去噪后的图像进行维纳滤波复原
result_wiener = adaptive_wiener(denoised, PSF, eps=1e-3)
graph.subplot(236)
graph.xlabel("Wiener deblurred after denoise")
graph.imshow(result_wiener, cmap='gray')

graph.show()

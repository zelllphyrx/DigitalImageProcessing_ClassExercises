import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import math
import cv2

# 仿真运动模糊函数
# 输入：图像尺寸 image_size, 运动角度 motion_angle
# 输出：归一化后的点扩散函数 PSF（Point Spread Function）
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)  # 初始化 PSF 为零矩阵
    print(image_size)  # 打印图像尺寸
    center_position = (image_size[0] - 1) / 2  # 计算中心位置
    print(center_position)

    # 计算运动角度的正切值和余切值
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan

    # 根据运动角度生成 PSF
    if slope_tan <= 1:  # 角度小于等于45度
        for i in range(35):  # 模拟运动的35个像素
            offset = round(i * slope_tan)  # 计算偏移量
            PSF[int(center_position + offset), int(center_position - offset)] = 1
    else:  # 角度大于45度
        for i in range(35):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1

    return PSF / PSF.sum()  # 归一化点扩散函数

# 对图片进行运动模糊
# 输入：原始图像 input，PSF，噪声参数 eps
# 输出：模糊后的图像
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 对图像进行二维傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps  # PSF 的傅里叶变换并添加噪声参数
    blurred = fft.ifft2(input_fft * PSF_fft)  # 模拟运动模糊的退化过程
    blurred = np.abs(fft.fftshift(blurred))  # 对结果进行中心化并取绝对值
    return blurred

# 逆滤波复原函数
# 输入：退化图像 input，PSF，噪声参数 eps
# 输出：逆滤波复原后的图像
def inverse(input, PSF, eps):
    input_fft = fft.fft2(input)  # 对退化图像进行傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps  # PSF 的傅里叶变换并添加噪声参数
    result = fft.ifft2(input_fft / PSF_fft)  # 逆滤波计算
    result = np.abs(fft.fftshift(result))  # 对结果进行中心化并取绝对值
    return result

# 维纳滤波复原函数
# 输入：退化图像 input，PSF，噪声参数 eps，维纳滤波参数 K
# 输出：维纳滤波复原后的图像
def wiener(input, PSF, eps, K=0.01):
    input_fft = fft.fft2(input)  # 对退化图像进行傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps  # PSF 的傅里叶变换并添加噪声参数
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)  # 计算维纳滤波器
    result = fft.ifft2(input_fft * PSF_fft_1)  # 应用维纳滤波器
    result = np.abs(fft.fftshift(result))  # 对结果进行中心化并取绝对值
    return result

# 加载并灰度化 Lena 图像
image = cv2.imread('C:/Users/Lenovo/Desktop/lena.jpg')  # 加载图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
img_h = image.shape[0]  # 图像高度
img_w = image.shape[1]  # 图像宽度

# 显示原始图像
graph.figure(1)
graph.xlabel("Original Image")
graph.gray()
graph.imshow(image)

# 生成运动模糊的 PSF
PSF = motion_process((img_h, img_w), 45)
blurred = np.abs(make_blurred(image, PSF, 1e-3))  # 对原始图像进行运动模糊

# 显示运动模糊图像
graph.subplot(231)
graph.xlabel("Motion blurred")
graph.imshow(blurred)

# 对运动模糊图像进行逆滤波复原
result = inverse(blurred, PSF, 1e-3)
graph.subplot(232)
graph.xlabel("Inverse deblurred")
graph.imshow(result)

# 对运动模糊图像进行维纳滤波复原
result = wiener(blurred, PSF, 0.0001)
graph.subplot(233)
graph.xlabel("Wiener deblurred (k=0.02)")
graph.imshow(result)

# 添加高斯噪声并显示退化图像
blurred_noisy = cv2.GaussianBlur(blurred, ksize=(0, 0), sigmaX=5, sigmaY=5)
graph.subplot(234)
graph.xlabel("Motion & Gaussian noisy blurred")
graph.imshow(blurred_noisy)

# 对退化图像（运动模糊+高斯噪声）进行逆滤波复原
result = inverse(blurred_noisy, PSF, 0.1 + 1e-3)
graph.subplot(235)
graph.xlabel("Inverse deblurred")
graph.imshow(result)

# 对退化图像（运动模糊+高斯噪声）进行维纳滤波复原
result = wiener(blurred_noisy, PSF, 0.1 + 0.0001)
graph.subplot(236)
graph.xlabel("Wiener deblurred (k=0.02)")
graph.imshow(result)

graph.show()

# 进行噪声去除
graph.figure(3)
graph.gray()
denoised = cv2.fastNlMeansDenoising(blurred_noisy.astype(np.uint8), None, 10, 7, 21)
graph.subplot(321)
graph.xlabel("Denoised Image")
graph.imshow(denoised)

# 对去噪后的图像进行逆滤波复原
result = inverse(denoised, PSF, 1e-3)
graph.subplot(322)
graph.xlabel("Inverse deblurred after denoising")
graph.imshow(result)

graph.show()

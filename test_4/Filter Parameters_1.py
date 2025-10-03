import cv2
import numpy as np


def calculate_energy_percentage(image, radius):
    """
    计算图像在给定半径范围内的能量百分比

    参数：
        image: 输入的图像（彩色图像）
        radius: 半径，指定频域内的能量计算范围

    返回：
        能量百分比
    """
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像的尺寸
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2  # 中心点的坐标

    # 对图像进行傅里叶变换
    f = np.fft.fft2(gray_image)  # 快速傅里叶变换
    fshift = np.fft.fftshift(f)  # 将低频移动到频谱中心
    magnitude_spectrum = np.abs(fshift) ** 2  # 计算功率谱（能量）

    # 计算图像的总能量
    total_energy = np.sum(magnitude_spectrum)

    # 创建一个掩膜(mask)，只保留指定半径范围内的能量
    mask = np.zeros((rows, cols), np.uint8)  # 初始化为全零的掩膜
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)  # 在掩膜上绘制圆形
    energy_within_radius = np.sum(magnitude_spectrum * mask)  # 计算掩膜内的能量

    # 计算能量百分比
    energy_percentage = (energy_within_radius / total_energy) * 100
    return energy_percentage


def find_radius_for_target_energy(image, target_energy_percentage, max_radius=300, step=1):
    """
    根据目标能量百分比，找到与之对应的半径 d0

    参数：
        image: 输入的图像
        target_energy_percentage: 目标能量百分比（例如90%、95%等）
        max_radius: 最大搜索半径，默认为300
        step: 半径的步长，默认为1

    返回：
        半径 d0 和实际能量百分比
    """
    # 逐步增加半径，计算对应的能量百分比
    for radius in range(1, max_radius, step):
        energy_percentage = calculate_energy_percentage(image, radius)
        if energy_percentage >= target_energy_percentage:
            return radius, energy_percentage  # 找到满足条件的半径，立即返回
    return None, None  # 如果搜索范围内没有满足条件的半径，返回None


# 加载图像
image = cv2.imread('C:\\Users\\Lenovo\\Desktop\\image_1.png')  # 替换为实际的图像路径

# 指定目标能量百分比
target_energy_percentage = 10.0  # 目标能量百分比（可以调整）

# 查找对应目标能量百分比的半径 d0
d0, actual_energy_percentage = find_radius_for_target_energy(image, target_energy_percentage)

if d0 is not None:
    # 输出找到的半径和对应的能量百分比
    print(f"半径 d0: {d0}, 达到的能量百分比: {actual_energy_percentage:.2f}%")
else:
    # 输出未找到半径的情况
    print("未能找到与目标能量百分比匹配的半径。")

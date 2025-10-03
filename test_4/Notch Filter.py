import cv2
import numpy as np

def fft(img):
    assert img.ndim == 2, 'img should be gray.'
    rows, cols = img.shape[:2]
    # 直接使用原始尺寸
    nimg = np.zeros((rows, cols))
    nimg[:rows, :cols] = img
    # 傅立叶变换
    fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 换位，将低频部分移到中间，高频部分移到四周
    return np.fft.fftshift(fft_mat)

def ifft(fft_mat):
    # 反换位，低频部分移到四周，高频部分移到中间
    f_ishift_mat = np.fft.ifftshift(fft_mat)
    # 傅立叶反变换
    img_back = cv2.idft(f_ishift_mat)
    # 将复数转换为幅度, sqrt(re^2 + im^2)
    img_back = cv2.magnitude(*cv2.split(img_back))
    # 标准化到0~255之间
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))

def notch_filter(img, h, w):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            mask[u,v]=0
    for u in range(rows):
        for v in range(cols):
            if abs(u - center[0]) < h and abs(v - center[1]) < w:
                mask[u, v] = 1

    return mask

# 读取图像并应用陷波滤波器
def apply_notch_filter(img_path, h, w):
    # 读取图像并转换为灰度图
    img_gray = cv2.imread(img_path, 0)
    # 获取图像尺寸
    rows, cols = img_gray.shape[:2]
    # 进行傅立叶变换
    fft_mat = fft(img_gray)
    # 创建陷波滤波器
    mask = notch_filter(img_gray, h, w)  # 这里传入图像本身，而不是 rows 和 cols
    # 确保 mask 的尺寸与 fft_mat 一致
    mask = np.resize(mask, fft_mat.shape)  # 调整 mask 的尺寸
    # 应用陷波滤波器
    filtered_fft = fft_mat * mask
    # 进行傅立叶反变换
    img_back = ifft(filtered_fft)
    # 显示结果
    cv2.imshow('Filtered Image', img_back)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 设置陷波滤波器参数
    h = 25  # 中心区域的高度
    w = 25  # 中心区域的宽度

    # 图像路径
    image_path = 'C:\\Users\\Lenovo\\Desktop\\1.png'

    # 应用陷波滤波器
    apply_notch_filter(image_path, h, w)

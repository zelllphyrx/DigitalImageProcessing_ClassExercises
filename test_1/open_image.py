# 导入OpenCV库
import cv2

# 读取指定路径下的图像
image = cv2.imread('C:/Users/Lenovo/Desktop/image_1.png')

# 多加一步检查图像是否正确读取
if image is None:
    print("图像文件读取失败，请检查路径是否正确")
else:
    # 显示图像
    cv2.imshow('Image', image)

    # 防止窗口立刻关闭
    cv2.waitKey(0)

    # 打印图像尺寸和指定点的RGB值
    print(f"图像尺寸: {image.shape}")
    print(f"图像中的某个像素值: {image[100, 100]}")

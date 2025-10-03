# 导入OpenCV库
import cv2

# 读取彩色图像
image = cv2.imread('C:/Users/Lenovo/Desktop/image_1.png')

# 先检查图像是否正确读取
if image is None:
    print("图像文件读取失败，请检查路径是否正确")
else:
    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示原图像
    cv2.imshow('Original Image', image)

    # 显示灰度图像
    cv2.imshow('Gray Image', gray_image)

    # 防止窗口立刻关闭并等待任意键按下关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

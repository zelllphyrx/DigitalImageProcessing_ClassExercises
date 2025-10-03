# 导入OpenCV库
import cv2

# 读取彩色图像
image = cv2.imread('C:/Users/Lenovo/Desktop/image_1.png')

# 检查图像是否正确读取
if image is None:
    print("图像文件读取失败，请检查路径是否正确")
else:
    # 首先将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 然后将灰度图像转换为二值图像
    # 参数解释gray_image-输入图像,127-阈值,255-阈值上限,cv2.THRESH_BINARY-二值化类型
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # 显示原图像
    cv2.imshow('Original Image', image)

    # 显示二值图像
    cv2.imshow('Binary Image', binary_image)

    # 防止窗口立刻关闭
    cv2.waitKey(0)
    cv2.destroyAllWindows()

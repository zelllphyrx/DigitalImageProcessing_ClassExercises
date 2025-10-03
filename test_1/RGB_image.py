# 导入OpenCV库
import cv2

# 从指定路径加载一张图片
image = cv2.imread('C:/Users/Lenovo/Desktop/image_1.png')

# 分离原图的三个通道
B, G, R = cv2.split(image)

# 创建三个空的与原图大小相同的单通道图像，分别用于显示R, G, B通道
zero_channel = B * 0

# 构建仅包含蓝色、绿色和红色通道的图像
blue_image = cv2.merge([B, zero_channel, zero_channel])
green_image = cv2.merge([zero_channel, G, zero_channel])
red_image = cv2.merge([zero_channel, zero_channel, R])

# 显示原图和三个分量图
cv2.imshow('Original Image', image)
cv2.imshow('Blue Channel', blue_image)
cv2.imshow('Green Channel', green_image)
cv2.imshow('Red Channel', red_image)

# 防止窗口立刻关闭
cv2.waitKey(0)

# 关闭所有打开的图像窗口
cv2.destroyAllWindows()

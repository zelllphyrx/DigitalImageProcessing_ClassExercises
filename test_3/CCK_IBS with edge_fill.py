import cv2
import numpy as np
import matplotlib.pyplot as plt


def convolution(k, p, data):
    n, m = data.shape
    padding = p // 2
    # 为输入图像增加边缘填充
    padded_data = cv2.copyMakeBorder(data, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    img_new = []

    for i in range(n):
        line = []
        for j in range(m):
            # 使用填充后的图像进行卷积
            a = padded_data[i:i + p, j:j + p]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)

    img_new = np.rint(img_new).astype('uint8')
    return np.array(img_new)


if __name__ == '__main__':
    img = cv2.imread('C:\\Users\\Lenovo\\Desktop\\image_1.png')
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k1 = np.array([
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9]
    ])

    k2 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    k3 = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])

    k4 = np.array([
        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]
    ])

    k5 = np.array([
        [1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49],
        [1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49],
        [1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49],
        [1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49],
        [1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49],
        [1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49],
        [1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49, 1 / 49]
    ])

    img_new__ = convolution(k1, 3, grayimg)
    img_new_2 = convolution(k2, 3, grayimg)
    img_new_3 = convolution(k3, 3, grayimg)
    img_new = convolution(k4, 5, grayimg)
    img_new_ = convolution(k5, 7, grayimg)

    cv2.imshow("img", img)
    cv2.imshow("grayimg", grayimg)
    cv2.imshow("k2 img", img_new_2)
    cv2.imshow("k3 img", img_new_3)

    # 使用灰度图与卷积结果的加权和
    dst2 = cv2.addWeighted(grayimg, 0.9, img_new_2, 0.1, 0)
    dst3 = cv2.addWeighted(grayimg, 0.9, img_new_3, 0.1, 0)
    cv2.imshow("dst2", dst2)
    cv2.imshow("dst3", dst3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

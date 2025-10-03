import numpy as np
import cv2

img = cv2.imread('C:\\Users\\Lenovo\\Desktop\\image_1.png')

grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fimg = np.fft.fft2(grayimg) #numpy.fft.fft2获取二维离散傅里叶变换频谱图
shiftfimg = np.fft.fftshift(fimg) #numpy.fft.fftshift进行零点移动获取相位图

#幅度谱
fspecturm = np.abs(shiftfimg)
normfspect = np.log(1+fspecturm) #尺度调整
normfspect = 255 * cv2.normalize(normfspect,0,1) #归一化

#相位谱
fphase = np.angle(shiftfimg)

#逆变换
imgback = np.fft.ifft2(fimg)
imgback = imgback.real #虚部只有残留，舍弃，只取实部

cv2.imshow("original",grayimg)
cv2.imshow("FFT specturm",normfspect)
cv2.imshow("FFT phase",fphase)
cv2.imshow("ifft",255 * cv2.normalize(imgback,0,1))

cv2.waitKey(0)

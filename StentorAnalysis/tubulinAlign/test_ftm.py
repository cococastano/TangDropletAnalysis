import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from FtmUtilities import *

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

# IMAGE FILEPATH

# ACTUALLY GOOD
# test_pic = "D:/(3) Kevin Stanford/Rotations Y2/Tang/data/180628 anti tubulin unwounded methanol/PSW/" \
#            "A_blue 10x nd4 p1s exp 2.tif"

# BAD 1
# test_pic = "D:/(3) Kevin Stanford/Rotations Y2/Tang/data/180629 anti tubulin 20ml-hr 70x250x100um 30cm tubing/" \
#            "blue 10x nd4 p1s exp 1.tif"

# BAD 2
# test_pic = "D:/(3) Kevin Stanford/Rotations Y2/Tang/data/180629 anti tubulin 20ml-hr 70x250x100um 30cm tubing/" \
#            "coverslip blue 20x nd4 p1s exp 5.tif"

# TEST
test_pic = "D:/(3) Kevin Stanford/Rotations Y2/Tang/test_img2.png"
# test_pic = "D:/(3) Kevin Stanford/Rotations Y2/Tang/data/test2.png"

# LOAD IMAGE
test = cv.imread(test_pic, cv.IMREAD_GRAYSCALE)
# test = cv.equalizeHist(test)
# test = cv.GaussianBlur(test, ksize=(5, 5), sigmaX=0)
# ret, test = cv.threshold(test, thresh=255*0.13, maxval=255, type=cv.THRESH_BINARY)
# test = cv.adaptiveThreshold(test,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
# ret2,test = cv.threshold(test,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# test = np.zeros((400, 400))
# spacer = np.arange(100, 300, 10)
#
# test[spacer, :] = 255

r, c = test.shape

hi = np.min((r, c))

cv.imshow('image', test)
cv.waitKey(0)

# 2D FFT -> Power spectrum
fourier = np.fft.fftshift(np.fft.fft2(test))
FS = np.abs(fourier)
PS = FS ** 2


# FTM orientation distribution
theta, ftm = FFT_orientation(PS, 5, 100)


# Anisotropy index
alpha = anisotropyIndex(theta, ftm)


print(alpha)
# Plot
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(theta, ftm)
plt.title('Unwounded')
plt.show()
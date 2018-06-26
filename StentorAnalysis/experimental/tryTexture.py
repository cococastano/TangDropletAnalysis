import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

# Check OpenCV loaded
print ("Open CV version " + cv.__version__)

# PARAMETERS ###########################################################################################################
# Source of trimmed pictures (N x N cut by finding dark regions, confirmed cell)
path = 'D:/(3) Kevin Stanford/Rotations Y2/Tang/test_openCV_droplets/trimmed_temp_0612'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

KSIZE = 11
morph_kernel = np.ones((5, 5))

# GABOR AND GLCM
for f in onlyfiles:
    # Grayscale original image
    filename = 'trimmed_temp_0612/' + f

    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    img = np.uint8(img)

    img = cv.equalizeHist(img)

    theta = np.array([1, 2, 3, 4]) * np.pi / 5

    for th in range(len(theta)):
        kernel = cv.getGaborKernel(ksize=(KSIZE,KSIZE), sigma=1, theta=theta[th], lambd=0.1, gamma=1)
        #print(kernel.shape)

        temp = cv.filter2D(img, ddepth=-1, kernel=kernel)
        temp = cv.equalizeHist(temp)

        # Filter

        ret, thresh2 = cv.threshold(temp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #thresh2 = cv.adaptiveThreshold(temp, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=225, C=3)
        thresh2 = cv.bitwise_not(thresh2)

        closing = cv.morphologyEx(thresh2, cv.MORPH_OPEN, morph_kernel)

        if th == 0:
            agg = np.copy(temp)
            agg_thresh = np.copy(thresh2)
            layer = np.copy(thresh2)
        else:
            agg = np.hstack((agg, temp))
            agg_thresh = np.hstack((agg_thresh, thresh2))
            layer = cv.bitwise_and(layer, thresh2)

    savename = f[:-4] + '_gabor.png'
    savename2 = f[:-4] + '_gaborThresh.png'
    savename3 = f[:-4] + '_gaborLayer.png'
    cv.imwrite(savename, agg)
    cv.imwrite(savename2, agg_thresh)
    cv.imwrite(savename3, layer)
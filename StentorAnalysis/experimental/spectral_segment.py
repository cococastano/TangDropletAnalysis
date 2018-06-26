import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

base = "D:/(3) Kevin Stanford/Rotations Y2/Tang/stentor_regen_clusters/edge_case_pics/"

pics = ["1", "2"]

kernel = np.ones((3, 3))

for p in pics:
    pic_name = base + p + '.bmp'

    raw = cv.imread(pic_name, cv.IMREAD_GRAYSCALE)
    raw = np.uint8(raw)
    raw = cv.equalizeHist(raw)

    img2 = cv.imread(pic_name, cv.IMREAD_COLOR)

    LOWEST_QUARTER = np.percentile(raw, 10)
    w = raw.shape[0]

    # Filter #####
    ret, thresh = cv.threshold(raw, thresh=256 * 0.25, maxval=255, type=cv.THRESH_BINARY)
    thresh2 = cv.adaptiveThreshold(raw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=225, C=3)
    thresh = cv.bitwise_not(thresh)
    thresh2 = cv.bitwise_not(thresh2)

    # Combine hard and adaptive threshold #####
    inter = cv.bitwise_and(thresh, thresh2)

    img2 = cv.bitwise_and(raw, inter)
    img2 = cv.equalizeHist(img2)

    edges = cv.Laplacian(inter, ddepth=cv.CV_16S, ksize=25)
    edges = np.absolute(edges)
    edges = np.uint8(edges)

    sharp = cv.subtract(img2, edges)

    # Distance transform
    dist = cv.distanceTransform(inter, distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_PRECISE)
    test = cv.normalize(dist, 0, 1., cv.NORM_MINMAX)

    print(np.max(test))

    cv.imshow('sharpened', test)
    cv.waitKey(0)
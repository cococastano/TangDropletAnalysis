import cv2 as cv
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
from utilities_shape_descriptor import *

# DESCRIPTION:
# From a database of trimmed, square pictures, find and save cell contours
# For each contour, analyze shape and save descriptors/features

# TODO: ~ 10-15% false positives need to be trimmed out, manually for now

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

# PARAMETERS ###########################################################################################################
# Source of trimmed pictures (N x N cut by finding dark regions, confirmed cell)
path = 'D:/(3) Kevin Stanford/Rotations Y2/Tang/test_openCV_droplets/quick_GC_class'
pic_root = 'quick_GC_class/'
save_root = 'drawn_pics_GC/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

# Image pre-processing parameters
KSIZE = 11
morph_kernel = np.ones((5, 5))

AREA_FLOOR = 100 * 10
AREA_CEIL = 200 * 200

num_iter = 2

# Output parameters
SAVE_SHAPE = 1      # (4) Basic shape features: eccentricity, circularity, solidity, convexity
SAVE_HU = 1         # (7) Hu moments
SAVE_TEXTURE = 0    # (14) Haralick texture features

N_OUTPUT = 2 + 4 * SAVE_SHAPE + 7 * SAVE_HU + 14 * SAVE_TEXTURE     # Allocate space for time point, filename, and data
data = np.zeros((len(onlyfiles) * 2, N_OUTPUT)) # Make 2 x as many rows as buffer for multiple cells in one image

# GET CONTOURS #########################################################################################################
cur_line = 0
next_mark = 0

for i in range(len(onlyfiles)):
    f = onlyfiles[i]
    # TODO: Better way to index time point (i.e. all times as XX.X format)
    if f[9] == '.':
        my_t = 0.5
    else:
        my_t = float(f[8])
    # my_t = i

    # Load image #####
    filename = pic_root + f
    print(f)

    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    img = np.uint8(img)
    img = cv.equalizeHist(img)

    img2 = cv.imread(filename, cv.IMREAD_COLOR)

    LOWEST_QUARTER = np.percentile(img, 10)
    w = img.shape[0]

    # Filter #####
    ret, thresh = cv.threshold(img, thresh=256 * 0.25, maxval=255, type=cv.THRESH_BINARY)
    thresh2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=225, C=3)
    thresh = cv.bitwise_not(thresh)
    thresh2 = cv.bitwise_not(thresh2)

    # Combine hard and adaptive threshold #####
    inter = cv.bitwise_and(thresh, thresh2)

    # Remove noise at edges #####
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, morph_kernel, iterations=num_iter)
    thresh2 = cv.morphologyEx(thresh2, cv.MORPH_OPEN, morph_kernel, iterations=num_iter)
    inter = cv.morphologyEx(inter, cv.MORPH_OPEN, morph_kernel, iterations=num_iter)

    # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, morph_kernel, iterations=num_iter)
    # thresh2 = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, morph_kernel, iterations=num_iter)
    # inter = cv.morphologyEx(inter, cv.MORPH_CLOSE, morph_kernel, iterations=num_iter)

    # Find contours #####
    _, contours, hierarchy = cv.findContours(inter, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Decision making #####
    for c in contours:
        A = cv.contourArea(c)
        M = cv.moments(c)

        # Area criteria
        if A >= AREA_FLOOR and A <= AREA_CEIL:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Intensity criteria
            mask = np.zeros_like(img)
            fill = cv.drawContours(mask, [c], 0, 255, cv.FILLED)

            mark = np.sum(fill & img)                           # Sum of actual pixel values in contour
            target = np.count_nonzero(fill) * LOWEST_QUARTER    # Sum if all pixels in contour were at 10th percentile

            if mark > target:
                continue

            # Picture border proximity criteria - skip if too close to edge (likely cutoff)
            if cx < 50 or cx > w - 50 or cy < 50 or cy > w - 50:
                continue

            draw_contour = np.copy(img2)
            cv.drawContours(draw_contour, [c], 0, (0, 255, 0), 2)

            savename = save_root + str(cur_line) + '.png'

            cv.imwrite(savename, draw_contour)

            data_toAdd = [my_t, cur_line]
            if SAVE_SHAPE:
                shape_names, shape_data = save_shape(c)
                data_toAdd = np.append(data_toAdd, shape_data)

            if SAVE_HU:
                hu_names, hu_data = save_hu(c)
                data_toAdd = np.append(data_toAdd, hu_data)

            data[cur_line, :] = data_toAdd
            cur_line += 1

    if 10 * (i + 1.0) / len(onlyfiles) >= next_mark:
        nearest = int(round(10 * (i + 1.0) / len(onlyfiles)))
        msg = nearest * '==' + (10 - nearest) * '--' + str(nearest * 10) + "% completed"
        print(msg)
        next_mark += 1

print('Found: ', cur_line)
data = data[0:cur_line, :]

names = ['Time', 'Picture Code']
if SAVE_SHAPE: names += shape_names
if SAVE_HU: names += hu_names

df = pd.DataFrame(data)
df.to_csv("GC_contour_data_updated.csv", header=names)

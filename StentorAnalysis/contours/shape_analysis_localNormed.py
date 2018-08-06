import cv2 as cv
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
from utilities_shape_analysis import *

########################################################################################################################
# From a database of trimmed, square pictures, find and save cell contours
# For each contour, analyze shape and save descriptors/features

# 20180806 Kevin Zhang
########################################################################################################################

# TODO: Integrate with droplet edge handling
# TODO: Currently ~50% false positive, but >95% of these cases are due to droplet edges or poor experimental conditions

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

# PARAMETERS ###########################################################################################################
# Source of trimmed pictures (N x N cut by finding dark regions, confirmed cell)
path = 'D:/(3) Kevin Stanford/Rotations Y2/Tang/test_openCV_droplets/trimmed_pics/'
save_root = '20180802_drawn_pics/'    # Root file to save pictures with drawn contours
csv_out = "20180802_cluster_data.csv" # File for CSV
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

# Image pre-processing parameters
# (OLD)
KSIZE = 11
morph_kernel = np.ones((5, 5))
num_iter = 2

THRESH = 125

AREA_FLOOR = 1000
AREA_CEIL = 200 * 200

# Output parameters
SAVE_SHAPE = 1      # (4) Basic shape features: eccentricity, circularity, solidity, convexity
SAVE_HU = 1         # (7) Hu moments
SAVE_TEXTURE = 1    # (14) Haralick texture features

N_OUTPUT = 2 + 4 * SAVE_SHAPE + 7 * SAVE_HU + 9 * SAVE_TEXTURE     # Allocate space for time point, filename, and data
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
    filename = path + f

    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(filename, cv.IMREAD_COLOR)
    img = np.uint8(img)

    w = img.shape[0]

    # Local norm
    gray = local_norm(img)

    # Threshold
    eq = cv.equalizeHist(gray)
    ret, thresh = cv.threshold(eq, THRESH, 255, cv.THRESH_BINARY)

    # Find contours #####
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Decision making #####
    final_contours = filter_contours(contours, img, AREA_FLOOR, AREA_CEIL)

    for c in final_contours:
        draw_contour = np.copy(img2)
        cv.drawContours(draw_contour, [c], 0, (0, 255, 0), 1)

        savename = save_root + str(cur_line) + '.png'
        cv.imwrite(savename, draw_contour)

        data_toAdd = [my_t, cur_line]
        if SAVE_SHAPE:
            shape_names, shape_data = save_shape(c)
            data_toAdd = np.append(data_toAdd, shape_data)

        if SAVE_HU:
            hu_names, hu_data = save_hu(c)
            data_toAdd = np.append(data_toAdd, hu_data)

        if SAVE_TEXTURE:
            texture_names, texture_data = save_texture(img, c)
            data_toAdd = np.append(data_toAdd, texture_data)

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
if SAVE_TEXTURE: names += texture_names

df = pd.DataFrame(data)
df.to_csv(csv_out, header=names)

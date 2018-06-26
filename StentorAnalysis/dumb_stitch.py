import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# DESCRIPTION:
# Dumb stitch to stack all partial images of a well at a specific time together.
# Does not perform any alignment of features
# Purely for visual overview of the entire well. Use individual images for actual image processing

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

dirpath = 'D:/(3) Kevin Stanford/Rotations Y2/Tang/data/20180512_8hrStentorDroplets/'

timepoints = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]

# Photos of a chip at each time point are arranged in 7 x 2 fashion, some overlap
CHIPDIM_X = 7
CHIPDIM_Y = 2

chips = ['A', 'B']

for i in range(len(chips)):
    chipname = chips[i]

    for t in range(len(timepoints)):
        current_dir = dirpath + 'T' + str(timepoints[t]) + '/'

        Y0 = []
        Y1 = []
        px_y, px_x = (0, 0)

        # Potentiall neeed to add an additional for loop if CHIPDIM_Y increases
        for x in range(CHIPDIM_X):
            leftname = current_dir + chipname + '_' + str(x) + '-1_T' + str(timepoints[t]) + '.bmp'
            rightname = current_dir + chipname + '_' + str(x) + '-0_T' + str(timepoints[t]) + '.bmp'

            left = cv.imread(leftname, cv.IMREAD_GRAYSCALE)
            right = cv.imread(rightname, cv.IMREAD_GRAYSCALE)

            if x == 0:
                Y0 = right
                Y1 = left

                # Size for trimming by PIXEL x, y
                # NOT using chip X, Y convention
                px_y, px_x = np.shape(left)

            else:
                # Trimming overlap in pixel y (CHIP X) direction for last image
                # if x == CHIPDIM_X - 1:
                #     # Trim by PIXEL x, y
                #     y_trim = int(px_y * 0.85)
                #
                #     left = left[0:y_trim, :]
                #     right = right[0:y_trim, :]

                Y0 = np.vstack((right, Y0))
                Y1 = np.vstack((left, Y1))

        # Trimming overlap in pixel x (CHIP Y) direction
        # x_trim = int(0.95 * px_x)
        #
        # Y1 = Y1[:, 0:x_trim]
        # Y0 = Y0[:, -x_trim:]

        img = np.hstack((Y1, Y0))

        # index = i * len(timepoints) + t
        # plt.figure(index)
        # plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title(chips[i] + 'T' + str(timepoints[t]))

        targetname = dirpath + chipname + '_T' + str(timepoints[t]) + '_full.bmp'
        cv.imwrite(targetname, img)

# plt.show()


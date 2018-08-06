import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve

########################################################################################################################
# Uses a fast convolution to identify "large" dark regions in a grayscale picture.
# Saves square around each dark region as a new image
# Options to allow manual labelling of images and to save rotated copies (for dataset enrichment)

# 20180806 Kevin Zhang
########################################################################################################################

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

##### RAW IMAGE FILE SOURCE #####
#TODO: Convert to os walk or similar so filename is not needed

dirpath = 'D:/(3) Kevin Stanford/Rotations Y2/Tang/data/20180602_8hrStentorDroplets/'
targetdir = 'D:/(3) Kevin Stanford/Rotations Y2/Tang/data/labelled_dataset_temp/'

trimdir = 'D:/(3) Kevin Stanford/Rotations Y2/Tang/test_openCV_droplets/trimmed_temp/'

timepoints = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]

# Photos of a chip at each time point are arranged in X by Y fashion, some overlap
CHIPDIM_X = 7
CHIPDIM_Y = 2

chips = ['A']#, 'B']

##### PROCESSING PARAMETERS #####
n = 50                                                          # Kernel size
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (n, n))     # Circular kernel to detect dark regions
max_error = 0.15                                                # Whitespace fraction tolerance
conv_limit = max_error * np.pi * (n * 0.50) ** 2                # Limit to decide a dark spot

w = 400                                                         # Size of square image to crop

AREA_FLOOR = 2000                                               # Criteria for identifying dark regions
AREA_CEIL = w ** 2 / 2

manual_save = True                                              # Whether to accept user input to label images

save_rotates = True                                             # Save rotated copies (for enriching CNN dataset)

M2 = cv.getRotationMatrix2D((w/2, w/2), 90, 1)                  # Rotation matrices for 90, 180, 270 deg. rotations
M3 = cv.getRotationMatrix2D((w/2, w/2), 180, 1)
M4 = cv.getRotationMatrix2D((w/2, w/2), 270, 1)


########################################################################################################################
for i in range(len(chips)):
    chipname = chips[i]

    for t in range(len(timepoints)):
        my_t = str(timepoints[t])

        current_dir = dirpath + 'T' + my_t + '/'

        num_pos = 0
        num_neg = 0
        num_mis = 0

        for x in range(CHIPDIM_X):
            for y in range(CHIPDIM_Y):
                my_x = str(x)
                my_y = str(y)

                # Filename generated for each image (Chip#, X, Y, T)
                name = current_dir + chipname + '-' + my_x + '-' + my_y + '_T' + my_t + '.bmp'

                # Load image
                original = cv.imread(name, cv.IMREAD_GRAYSCALE)
                original = np.uint8(original)

                img_color = cv.imread(name, cv.IMREAD_COLOR)
                img = np.copy(original)

                (max_y, max_x) = np.shape(img)

                # Threshold 0/1, then convolution to find dark spots
                thresh = cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=225, C=3)

                conv = fftconvolve(thresh, kernel, mode='same')
                conv[conv < conv_limit] = 0
                conv[conv >= conv_limit] = 255
                conv = np.uint8(conv)

                # Set borders to white to avoid contours combining
                conv[0, :] = conv[-1, :] = conv[:, 0] = conv[:, -1] = 255

                # Find contours (unique dark spots)
                img2, contours, hierarchy = cv.findContours(conv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                if contours is None:
                    continue

                # Draw square around, pay attention to limits
                for c in contours:
                    M = cv.moments(c)
                    if M['m00'] == 0 or cv.contourArea(c) > AREA_CEIL or cv.contourArea(c) < AREA_FLOOR:
                        continue

                    cv.drawContours(img_color, [c], 0, (0, 255, 0), 3)

                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    pad_dist = w / 2
                    x0 = cx - pad_dist
                    x1 = cx + pad_dist - 1
                    y0 = cy - pad_dist
                    y1 = cy + pad_dist - 1

                    if x0 < 0:
                        x0 = 0
                        x1 += np.abs(cx - pad_dist)
                    elif x1 >= max_x:
                        x0 -= x1 - (max_x - 1)
                        x1 = max_x - 1

                    if y0 < 0:
                        y0 = 0
                        y1 += np.abs(cy - pad_dist)
                    elif y1 >= max_y:
                        y0 -= y1 - (max_y - 1)
                        y1 = max_y - 1

                    # Extract region of interest from original image and rotate 3 times
                    ROI_1 = original[y0:y1 + 1, x0:x1 + 1]
                    ROI_2 = cv.warpAffine(ROI_1, M2, (w, w))
                    ROI_3 = cv.warpAffine(ROI_1, M3, (w, w))
                    ROI_4 = cv.warpAffine(ROI_1, M4, (w, w))


                    # Label images depending on manual_save flag
                    if manual_save:
                        cv.imshow('Contains cell? (Y)es, (N)o, (S)kip, (Q)uit', ROI_1)
                        k = cv.waitKey(0) & 0xFF
                        if k == ord('y'):       # Positive case (Stentor)
                            savebase = targetdir + '1_T' + my_t + '_' + chipname + '_' + str(num_pos)

                            cv.imwrite(trimdir + '/T' + my_t + '_' + str(num_pos) + '.png', ROI_1)

                            num_pos += 1
                            cv.destroyAllWindows()

                        elif k == ord('n'):     # Negative case (No Stentor)
                            savebase = targetdir + '0_T' + my_t + '_' + chipname + '_' + str(num_neg)

                            num_neg += 1
                            cv.destroyAllWindows()

                        elif k == ord('s'):     # Bad image (Discard)
                            cv.destroyAllWindows()
                            continue

                        elif k == ord('q'):
                            cv.destroyAllWindows()
                            sys.exit('User quit image labelling')

                        else:
                            savebase = targetdir + 'mislabelled_T' + my_t + '_' + chipname + '_' + str(num_mis)
                            num_mis += 1
                            cv.destroyAllWindows()

                    else:
                        savebase = 'T' + my_t + '_' + chipname + '_' + str(num_pos)
                        num_pos += 1

                    # Save images
                    cv.imwrite(savebase + '_r1.png', ROI_1)

                    if save_rotates:
                        cv.imwrite(savebase + '_r2.png', ROI_2)
                        cv.imwrite(savebase + '_r3.png', ROI_3)
                        cv.imwrite(savebase + '_r4.png', ROI_4)

        print('Finished chip ' + chipname + ' at Time ' + my_t)

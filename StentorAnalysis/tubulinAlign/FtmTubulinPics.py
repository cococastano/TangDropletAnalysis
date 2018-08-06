import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from FtmUtilities import *

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

# IMAGE FILEPATH
master = "D:\\(3) Kevin Stanford\\Rotations Y2\\Tang\\data\\20180801_trimmed_tubulin\\"

dir = ["Wounded", "Unwounded"]

for root, dirs, files in os.walk(master, topdown=False):

    for name in dirs:
        print(name)
        curdir = os.path.join(root, name)

        for _, _, files in os.walk(curdir):

            for name in files:
                if name[-3:] != 'png':
                    continue

                file = os.path.join(curdir, name)
                print(file)
                # LOAD IMAGE
                test = cv.imread(file, cv.IMREAD_GRAYSCALE)
                cv.imshow('image', test)
                cv.waitKey(100)

                # 2D FFT -> Power spectrum
                fourier = np.fft.fftshift(np.fft.fft2(test))
                FS = np.abs(fourier)
                PS = FS ** 2

                # FTM orientation distribution
                theta, ftm = FFT_orientation(PS, 5, 20)

                # Anisotropy index
                alpha = anisotropyIndex(theta, ftm)

                print("Anisotropy index: ", alpha)
                # Plot
                plt.figure()
                ax = plt.subplot(111, projection='polar')
                ax.plot(theta, ftm)
                plt.title(name)

plt.show()
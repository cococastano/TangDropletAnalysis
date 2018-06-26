import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

# Parameters ###########################################################################################################
AREA_FLOOR = 100 * 10
AREA_CEIL = 200 * 200

# TODO Write code to convolve, find large dark regions, draw large square around it, and use as input here
# TODO (Once more images are available) Turn into a for loop / function
filename = 'pics/trim2.bmp'

k = 5                       # Size for Gaussian
kernel = np.ones((3, 3))    # Kernel for morphological transform

# Load image ###########################################################################################################
# Grayscale original image
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
img = np.uint8(img)

# Average value to use for smearing droplet edges
AVERAGE = np.mean(img)

# Color version for drawing contours
img2 = cv.imread(filename, cv.IMREAD_COLOR)

# Pre-processing #######################################################################################################
# Gaussian
gs = cv.GaussianBlur(img, (k, k), sigmaX=0)

# # Laplace
# lp = cv.Laplacian(img, ddepth=cv.CV_64F, ksize=3)
# lp = np.absolute(lp)
# lp = np.uint8(lp)
# lp = cv.bitwise_not(lp)

# Implement a high pass filter
hp_gaussianKernel = cv.getGaussianKernel(ksize=k, sigma=0, ktype=cv.CV_64F)

hp_mat = np.dot(hp_gaussianKernel, hp_gaussianKernel.T)
hp_mat = hp_mat * -1
hp_mat[k / 2, k / 2] += 1
# print(hp_mat)

hp_img = cv.filter2D(img, ddepth=cv.CV_8U, kernel=hp_mat)
hp_img_inv = cv.bitwise_not(hp_img)
hp_img_inv = np.uint8(hp_img_inv * 0.5)

# Edge sharpened is original image + scaled HP image
sharp = cv.add(img, hp_img_inv)
sharp_eq = cv.equalizeHist(sharp)

# Thresholding #########################################################################################################

# Adaptive thresholding
thresh_adapt = cv.adaptiveThreshold(sharp_eq, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=225, C=3)

# Hist eq on only dark regions
dark = cv.bitwise_not(thresh_adapt)
iso = cv.bitwise_and(img, dark)
iso = cv.equalizeHist(iso)

# Combine and use automatic binary threshold
comb = cv.bitwise_or(iso, thresh_adapt)
ret, thresh2 = cv.threshold(comb,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
thresh2 = cv.bitwise_not(thresh2)

#holes_removed = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, kernel, iterations=5)

# Finding Cell #########################################################################################################
# Distance transform
dist = cv.distanceTransform(thresh2, distanceType=cv.DIST_L2, maskSize=5)

# Eliminate main bodies, keeping pixels close to edges
droplets = np.copy(thresh2)
droplets[dist > 1] = 0
# droplets[dist > 3] = 0

# Hough transform to find circles
# Allow some overlap (imperfect droplet circles, shadows, etc.)
# TODO Approach from both sides to find optimal radius cutoffs
circles = cv.HoughCircles(droplets, cv.HOUGH_GRADIENT, 1, minDist=20, param1=60, param2=30, minRadius=100, maxRadius=150)

blank = np.zeros_like(thresh2)
hough_circles = np.copy(img2)

# TODO Need a condition where if no Hough circles are found, we take the entire cropped droplet region
if circles is not None:
    circles = np.uint16(np.around(circles))
    for j in circles[0, :]:
        # Filled circle to mask droplet region
        cv.circle(blank, (j[0], j[1]), j[2], 255, -1)
        # White circle to erase droplet edge
        cv.circle(gs, (j[0], j[1]), j[2], AVERAGE, 2)

        # Color circle on original image
        cv.circle(hough_circles, (j[0], j[1]), j[2], (0, 255, 0), 3)
        cv.circle(hough_circles, (j[0], j[1]), 2, (0, 0, 255), 3)

# Use filled Hough circles as a mask, threshold
droplet_interior = blank & gs
droplet_interior = cv.add(droplet_interior, cv.bitwise_not(blank))
ret, thresh_final = cv.threshold(droplet_interior,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# thresh_final = cv.adaptiveThreshold(droplet_interior, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=99, C=3)

# Sometimes Hough circle cuts across cell. Use distance metric to partially fix
thresh_final[dist > 3] = 0

# Dilation to fix remaining droplet edges, self-overlapping contours
thresh_final = cv.dilate(thresh_final, kernel=kernel, iterations=2)

# Draw all contours
im2, contours, hierarchy = cv.findContours(thresh_final, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
im2 = cv.drawContours(np.zeros_like(img2), contours, -1, (0, 255, 0), 1)
final_contours = np.copy(img2)

# Find cell contours using area
for c in contours:
    M = cv.moments(c)
    if cv.contourArea(c) >= AREA_FLOOR and cv.contourArea(c) <= AREA_CEIL:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # if img[cy, cx] <= ret2:
        #     blank = cv.drawContours(blank, [c], 0, (0, 255, 0), 3)

        final_contours = cv.drawContours(final_contours, [c], 0, (0, 255, 0), 2)


# Plots ################################################################################################################
plt.figure(1)
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Original')
plt.axis('off')
plt.subplot(2, 3, 2), plt.imshow(droplets, cmap='gray', vmin=0, vmax=255), plt.title('Filtered + dist. transform for Hough circles')
plt.axis('off')
plt.subplot(2, 3, 3), plt.imshow(hough_circles, cmap='brg', vmin=0, vmax=255), plt.title('Hough Circles')
plt.axis('off')
plt.subplot(2, 3, 4), plt.imshow(droplet_interior, cmap='gray', vmin=0, vmax=255), plt.title('Smeared droplet edges')
plt.axis('off')
plt.subplot(2, 3, 5), plt.imshow(thresh_final, cmap='gray', vmin=0, vmax=255), plt.title('Fixed cut cell then Threshold')
plt.axis('off')
plt.subplot(2, 3, 6), plt.imshow(im2, cmap='brg', vmin=0, vmax=255), plt.title('All Contours')
plt.axis('off')

plt.figure(2)
plt.imshow(final_contours, cmap='brg')

plt.show()

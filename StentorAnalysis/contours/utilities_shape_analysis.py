import cv2 as cv
import numpy as np

from skimage.feature import greycomatrix

########################################################################################################################
# Contains utilities for image processing, contour selection, and feature extraction from contours

# 20180806 Kevin Zhang
########################################################################################################################'

# Performs local normalization of a grayscale image
def local_norm(img):
    gray = np.uint8(img)
    float_gray = gray / 255.0

    # Subtract mean
    blur = cv.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur

    # Divide by variance
    blur = cv.GaussianBlur(num ** 2, (0, 0), sigmaX=20, sigmaY=20)
    den = cv.pow(blur, 0.5)

    gray = num / den

    # Rescale
    gray = cv.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
    gray *= 255
    gray = np.uint8(gray)

    return gray

# Selects contours meeting the following criteria:
# 1 - Area between area floor and ceiling
# 2 - Center not within [PAD] pixels of borders
# 3 - ROI in original image is relatively dark
# 4 - Contour is not entirely enclosed by another contour
def filter_contours(contours, img, floor, ceiling):
    pad = 100                       # Padding size to check centroid location
    low20 = np.percentile(img, 20)  # 10th percentile of original image to check intensity

    contours = np.array(contours)
    w = img.shape[0]

    # First pass to eliminate contours by area
    mask = np.zeros((len(contours),), dtype=bool)
    areas = [cv.contourArea(c) for c in contours]
    cut = [floor < a < ceiling for a in areas]

    mask[cut] = True
    mask = np.array(mask)

    remaining_contours = contours[mask]

    # Second pass to check centroid location, total intensity, and parent
    mask = np.ones((len(remaining_contours),), dtype=bool)

    for i in range(len(remaining_contours)):
        c = remaining_contours[i]

        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Centroid check
        if cx < pad or cx > w - pad or cy < pad or cy > w - pad:
            mask[i] = False
            continue

        # Intensity check
        blank = np.zeros_like(img)
        cv.drawContours(blank, [c], 0, 255, cv.FILLED)

        mark = np.sum(blank & img)  # Sum of actual pixel values in contour
        target = np.count_nonzero(blank) * low20  # Sum assuming all pixels in contour at 10th percentile

        if mark > target:
            mask[i] = False
            continue

        # Parent-child check
        template = np.zeros_like(img)
        cv.drawContours(template, [c], 0, 255, -1)

        for j in range(len(remaining_contours)):
            if i == j:
                continue
            child = remaining_contours[j]
            blank = np.zeros_like(img)
            cv.drawContours(blank, [child], 0, 255, -1)

            if np.all(cv.bitwise_or(blank, template) == template):  # Means contour j is indeed a child
                mask[j] = False

    remaining_contours = remaining_contours[mask]

    return remaining_contours

# Save 4 basic geometric shape descriptors from a contour c
def save_shape(c):
    if c is None:
        print('No contour provided')
        return []

    area = cv.contourArea(c)
    peri = cv.arcLength(c, closed=True)
    minRect = cv.minAreaRect(c)
    size = minRect[1]
    h = max(size)
    w = min(size)

    hull = cv.convexHull(c)
    hull_area = cv.contourArea(hull)
    hull_peri = cv.arcLength(hull, closed=True)

    ECC = float(w) / h
    CIRC = 4 * np.pi * area / peri ** 2
    SOLID = float(area) / hull_area
    CONV = float(hull_peri) / peri

    names = ['Eccentricity', 'Circularity', 'Solidity', 'Convexity']
    data = [ECC, CIRC, SOLID, CONV]

    return names, data

# Save 7 Hu moments from a contour c
def save_hu(c):
    if c is None:
        print('No contour provided')
        return []

    M = cv.moments(c)

    names = ['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7']
    data = cv.HuMoments(M)

    return names, data

# Save 14 Haralick texture features from a contour c TODO only implemented 9 so far
def save_texture(image, c):
    # Uses contour as a mask to define a region from image to extract texture data from

    nd = 1  # Neighbor distance
    angles = [0.00 * np.pi, 0.25 * np.pi, 0.50 * np.pi, 0.75 * np.pi]  # Four axes

    # Generate mask of cell region
    mask = np.zeros_like(image)
    cv.drawContours(mask, [c], 0, 255, cv.FILLED)

    new = cv.bitwise_and(mask, image)

    # Get GLCM of cell region only by removing all 0-0 pairs that appear in GLCM of mask
    glcm = greycomatrix(image=new, distances=[nd], angles=angles, levels=256, symmetric=True)
    glcm_correct_mask = greycomatrix(mask, distances=[nd], angles=angles, levels=256, symmetric=True)

    glcm[0, 0, :, :] -= glcm_correct_mask[0, 0, :, :]

    # Useful indexes
    i = np.arange(1, 257).reshape((256, 1))
    j = i.T

    plus = j + i
    minus = np.abs(j - i)

    # Find texture features
    f = np.zeros((9, 1))
    for a in range(len(angles)):
        raw_glcm = glcm[:, :, 0, a]

        if np.sum(raw_glcm) == 0:
            p = raw_glcm
            print('Error: Zero sum GLCM')
        else:
            p = raw_glcm / np.sum(raw_glcm)

        u = np.sum(p * i)
        s = np.sum(p * (i - u) ** 2)

        # ENERGY
        f[0] += np.sum(p ** 2)

        # ENTROPY
        f[1] -= np.sum(p * np.log(p + 10e-10))

        # CORRELATION
        f[2] += np.sum((j - u) * (i - u) * p) / (s ** 2)

        # INVERSE DIFFERENCE MOMENT
        f[3] += np.sum(p / (1 + minus ** 2))

        # INERTIA
        f[4] += np.sum(minus ** 2 * p)

        # SUM AVERAGE
        for n in range(2, 256 * 2 + 1):
            p_plus = np.sum(p[plus == n])

            f[5] += n * p_plus

        # CLUSTER SHADE
        f[6] += np.sum((plus - 2 * u) ** 3 * p)

        # CLUSTER PROMINENCE
        f[7] += np.sum((plus - 2 * u) ** 4 * p)

        # HARALICK'S CORRELATION
        f[8] += np.sum(np.dot(i, j) * p - u ** 2) / s ** 2

    # Average across all axes to approximate rotational invariance
    f = f / 4

    names = ['Energy', 'Entropy', 'Correlation', 'Inv. Diff. Moment', 'Inertia', 'Sum Average', 'Cluster Shade',
             'Cluster Prominence', 'Haralick\'s Correlation']

    return names, f.T

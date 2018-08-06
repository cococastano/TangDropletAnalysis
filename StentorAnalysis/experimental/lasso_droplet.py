import numpy as np
import cv2 as cv

from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

base = "D:/(3) Kevin Stanford/Rotations Y2/Tang/stentor_regen_clusters/edge_case_pics/"

pics = range(17)

AREA_FLOOR = 200 * 200 * 0.05
AREA_CEILING = 200 * 200 * 100

n = 50
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (n, n))     # Circular kernel to detect dark regions
max_error = 0.60                                                # Whitespace fraction tolerance
conv_limit = 255 * max_error * np.pi * (n * 0.50) ** 2          # Limit to decide a dark spot

morph_kernel = np.ones((3, 3))
num_iter = 2

for p in pics:
    try:
        pic_name = base + str(p) + '.bmp'
        raw = cv.imread(pic_name, cv.IMREAD_GRAYSCALE)
        raw = np.uint8(raw)
    except:
        pic_name = base + str(p) + '.png'
        raw = cv.imread(pic_name, cv.IMREAD_GRAYSCALE)
        raw = np.uint8(raw)

    # Pre hist eq copy
    raw_copy = np.copy(raw)

    raw = cv.equalizeHist(raw)

    img2 = cv.imread(pic_name, cv.IMREAD_COLOR)

    w = raw.shape[0]

    # Filter #####

    ret, def_edge = cv.threshold(raw, thresh=256 * 0.15, maxval=255, type=cv.THRESH_BINARY)

    ret, max_edge = cv.threshold(raw, thresh=256 * 0.25, maxval=255, type=cv.THRESH_BINARY)

    def_edge = cv.bitwise_not(def_edge)
    def_edge_copy = np.copy(def_edge)
    max_edge = cv.bitwise_not(max_edge)

    # Get rid of bulk of white blobs (cells)
    conv = fftconvolve(def_edge, kernel, mode='same')
    conv[conv < conv_limit] = 255
    conv[conv >= conv_limit] = 0
    conv = np.uint8(conv)
    conv = cv.dilate(conv, np.ones((3, 3)), iterations=2)

    cell_cand = cv.bitwise_not(conv)
    cell_cand = cv.dilate(cell_cand, np.ones((3, 3)), iterations=5)

    _, contours, hierarchy = cv.findContours(cell_cand, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for c in contours:
        minRect = cv.minAreaRect(c)
        size_rect = minRect[1]
        h_rect = max(size_rect)
        w_rect = min(size_rect)

        if h_rect / w_rect >= 5.0:
            cv.drawContours(cell_cand, [c], 0, 0, cv.FILLED)

    def_edge = cv.bitwise_and(def_edge, conv)
    # cv.imwrite('def_edge.png', def_edge)

    # # Single pixel link
    # Straight
    k1 = np.array([[0, 0, 0],
                   [1, 0, 1],
                   [0, 0, 0]])

    k2 = k1.T

    k3 = np.eye(3, 3)

    k4 = np.flip(k3, 1)

    # # Knight's moves
    k5 = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, 0, 0]])
    k6 = k5.T
    k7 = np.flip(k5, 1)
    k8 = np.flip(k6, 1)

    k9 =  np.flip(k5, 0)
    k10 = np.flip(k6, 0)
    k11 = np.flip(k7, 0)
    k12 = np.flip(k8, 0)

    # Fixer masks
    fm1 = np.array(fftconvolve(def_edge, k1, mode='same') > 500)
    fm2 = np.array(fftconvolve(def_edge, k2, mode='same') > 500)
    fm3 = np.array(fftconvolve(def_edge, k3, mode='same') > 500)
    fm4 = np.array(fftconvolve(def_edge, k4, mode='same') > 500)
    fm5 = np.array(fftconvolve(def_edge, k5, mode='same') > 500)
    fm6 = np.array(fftconvolve(def_edge, k6, mode='same') > 500)
    fm7 = np.array(fftconvolve(def_edge, k7, mode='same') > 500)
    fm8 = np.array(fftconvolve(def_edge, k8, mode='same') > 500)
    fm9 = np.array(fftconvolve(def_edge, k9, mode='same') > 500)
    fm10 = np.array(fftconvolve(def_edge, k10, mode='same') > 500)
    fm11 = np.array(fftconvolve(def_edge, k11, mode='same') > 500)
    fm12 = np.array(fftconvolve(def_edge, k12, mode='same') > 500)

    temp = np.copy(def_edge)
    temp[fm1 | fm2 | fm3 | fm4 | fm5 | fm6 | fm7 | fm8 | fm9 | fm10 | fm11 | fm12] = 255

    # cv.imwrite('single_link.png', temp)

    # Find lines and then contours 2x

    lines = cv.HoughLinesP(temp, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=8)
    if lines is not None:
        for ell in range(len(lines)):
            for x1, y1, x2, y2 in lines[ell]:
                blank = np.zeros_like(temp)
                cv.line(blank, (x1, y1), (x2, y2), 255, 2)

                if np.sum(cv.bitwise_and(blank, cell_cand)) > 0:
                    continue
                else:
                    pass
                    #cv.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Pad borders with white
    maxx = temp.shape[1]
    maxy = temp.shape[0]

    temp[:, 0] = temp[:, maxx-1] = temp[0, :] = temp[maxy-1, :] = 255
    temp = cv.dilate(temp, np.ones((3, 3)), iterations=1)

    _, contours, hierarchy = cv.findContours(temp, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    padded = cv.dilate(temp, kernel=np.ones((3,3)), iterations=3)
    padded[:, 0] = padded[:, maxx - 1] = padded[0, :] = padded[maxy - 1, :] = 0

    blank3 = np.zeros_like(temp)

    for c in contours:
        A = cv.contourArea(c)
        M = cv.moments(c)
        P = cv.arcLength(c, closed=True)

        CIRC = 4 * np.pi * A / P ** 2

        if A <= 200:
            cv.drawContours(temp, [c], 0, 0, cv.FILLED)

        # Area criteria
        if A >= AREA_FLOOR and A <= AREA_CEILING:
            hull = cv.convexHull(c)
            hull_a = cv.contourArea(hull)
            hull_p = cv.arcLength(hull, closed=True)

            CIRC = 4 * np.pi * hull_a / hull_p ** 2

            blank = np.zeros_like(temp)
            cv.drawContours(blank, [hull], 0, 255, cv.FILLED)

            # How much of candidate cell does it cover
            coverage = np.sum(cv.bitwise_and(blank, cell_cand))

            # How many "crossings"
            blank2 = np.zeros_like(temp)
            cv.drawContours(blank2, [hull], 0, 255, 1)

            crossings = cv.bitwise_and(padded, blank2)

            _, crossing_contours, hierarchy = cv.findContours(crossings, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            cv.drawContours(padded, [hull], 0, 150, 2)
            #cv.imwrite('check_contours.png', padded)

            if CIRC >= 0.70 and coverage > 0.50 * np.sum(cell_cand) and len(crossing_contours) < 3:
                blank3 = np.zeros_like(temp)
                blank3[:, 0] = blank3[:, maxx - 1] = blank3[0, :] = blank3[maxy - 1, :] = 255

                if np.sum(blank3) == np.sum(cv.bitwise_and(blank3, blank2)):
                    continue

                cv.drawContours(temp, [hull], 0, 0, cv.FILLED)

                # TODO write as a function
                # draw a second convex hull using new partial filled in image. Only draw if area increase is not too high


                _, contours_second, hierarchy = cv.findContours(temp, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

                for c2 in contours_second:
                    A = cv.contourArea(c2)

                    # Area criteria
                    if A >= AREA_FLOOR and A <= AREA_CEILING:
                        hull = cv.convexHull(c2)
                        hull_a = cv.contourArea(hull)
                        hull_p = cv.arcLength(hull, closed=True)

                        CIRC = 4 * np.pi * hull_a / hull_p ** 2

                        blank = np.zeros_like(temp)
                        cv.drawContours(blank, [hull], 0, 255, cv.FILLED)

                        # How much of candidate cell does it cover
                        coverage = np.sum(cv.bitwise_and(blank, cell_cand))

                        # How many "crossings"
                        blank2 = np.zeros_like(temp)
                        cv.drawContours(blank2, [hull], 0, 255, 1)

                        crossings = cv.bitwise_and(padded, blank2)

                        _, crossing_contours, hierarchy = cv.findContours(crossings, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

                        #cv.drawContours(padded, crossing_contours, -1, 150, 2)

                        if CIRC >= 0.70 and coverage > 0.50 * np.sum(cell_cand) and len(crossing_contours) < 3:
                            blank3 = np.zeros_like(temp)
                            blank3[:, 0] = blank3[:, maxx - 1] = blank3[0, :] = blank3[maxy - 1, :] = 255

                            if np.sum(blank3) == np.sum(cv.bitwise_and(blank3, blank2)):
                                continue

                            # Draw a second filled convex hull, but shrunk to avoid erasing too much droplet edge
                            blank3[:, 0] = blank3[:, maxx - 1] = blank3[0, :] = blank3[maxy - 1, :] = 0
                            cv.drawContours(blank3, [hull], 0, 255, cv.FILLED)
                            cv.drawContours(img2, [hull], 0, (0, 0, 255), 2)


    blank3 = cv.bitwise_not(blank3)
    #slightly shrink second hull to avoid erasing too much edge
    blank3 = cv.dilate(blank3, kernel=np.ones((3,3)), iterations=1)

    temp2 = cv.bitwise_and(temp, blank3)
    #temp2 = cv.dilate(temp2, kernel=np.ones((3,3)), iterations=3)

    # cv.imwrite('final_edges.png', temp2)

    raw_copy = cv.equalizeHist(raw_copy)
    #raw_copy[raw_copy > 100] = 200
    raw_copy[temp2 == 255] = 255
    #raw_copy[temp2 != 255] += 50
    #raw_copy = cv.equalizeHist(raw_copy)

    max_temp = cv.dilate(temp2, kernel=np.ones((3,3)), iterations=5)
    diff = cv.subtract(max_temp, temp2)
    shadow = cv.bitwise_and(diff, max_edge)
    shadow = cv.bitwise_and(shadow, cv.bitwise_not(def_edge_copy))


    final_mask = cv.bitwise_or(shadow, temp2)
    #final_mask = cv.erode(final_mask, kernel=np.ones((3,3)), iterations=1)

    #raw_copy[final_mask == 255] = 255


    LOWEST_QUARTER = np.percentile(raw_copy, 20)

    edges = cv.Laplacian(raw_copy, ddepth=cv.CV_16S, ksize=3)
    edges = np.abs(edges)
    edges = np.uint8(edges)
    edges = cv.equalizeHist(edges)

    test = cv.add(edges, raw_copy)
    test = cv.GaussianBlur(test, ksize=(5,5), sigmaX=0)

    ret, thresh = cv.threshold(raw_copy, thresh=255*0.20, maxval=255, type=cv.THRESH_BINARY)
    thresh2 = cv.adaptiveThreshold(raw_copy, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=225, C=3)
    thresh = cv.bitwise_not(thresh)
    thresh2 = cv.bitwise_not(thresh2)

    #
    # # Combine hard and adaptive threshold #####
    inter = cv.bitwise_and(thresh, thresh2)

    inter[final_mask == 255] = 0
    #
    # Remove noise at edges #####
    #inter = cv.morphologyEx(inter, cv.MORPH_OPEN, np.ones((3,3)), iterations=1)
    #

    # _, drop_cont, drop_heir = cv.findContours(temp2, cv.RETR_TREE< cv.CHAIN_APPROX_NONE)
    _, cell_cont, cell_heir = cv.findContours(inter, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    area = [cv.contourArea(c) for c in cell_cont]

    max_area_idx = np.argmax(area)

    cv.drawContours(img2, [cell_cont[max_area_idx]], 0, (255, 0, 0), 2)
    img2[temp2 == 255, :] = (0, 0, 255)

    cv.imshow('no fix', img2)
    # cv.imwrite(str(p) + "_erased.png", raw_copy)
    cv.waitKey(0)

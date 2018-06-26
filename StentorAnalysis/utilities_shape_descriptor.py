import cv2 as cv
import numpy as np

# Utilities for extracting shape data from contours

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

def save_hu(c):
    if c is None:
        print('No contour provided')
        return []

    M = cv.moments(c)

    names = ['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7']
    data = cv.HuMoments(M)

    return names, data



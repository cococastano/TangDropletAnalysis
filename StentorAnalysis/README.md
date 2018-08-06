### StentorAnalysis ###

TODO: More detailed descriptions

## Typical pipeline: ##
# contours #
Process raw images
(1) dark_candidates.py crops and stores likely cell regions to a square.

(2) shape_analysis.py or shape_analysis_localNormed.py to read square images, find contours, and save cell data. See utilities_shape_analysis.py for functions used.

# learningRegen
Learning regeneration stages from contour and/or image data
(3) cluster.py to perform unsupervised clustering

## tubulinAlign ##
Uses Fourier Transform Method to quantify orientation and anisotropy index [0, 1] of anti-tubulin stains. 
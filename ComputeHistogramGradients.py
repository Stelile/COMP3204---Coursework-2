import numpy as np
from skimage.feature import hog

def compute_HOG(img_hsv):
    if len(img_hsv.shape) == 3:
        intensity_img = img_hsv[:, :, 2]
    else:
        intensity_img = img_hsv

    hog_features = hog(intensity_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)

    return hog_features
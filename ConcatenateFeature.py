import numpy as np
import cv2
from LocalBinaryPatterns import extract_lbp
from ComputeHistogramGradients import compute_HOG
from glcm import glcm_features
from ColorHistogram import extract_color_features

def extract_feature(image: np.ndarray, mask=None) -> np.ndarray:
    if image.ndim == 3:
        lbp_input = image[:, :, 2]
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bgr_input = image.astype(np.uint8, copy=False)
    else:
        lbp_input = image
        grey = image
        bgr_input = cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)

    _, lbp_histogram = extract_lbp(lbp_input, mask=mask)
    hog_feature = compute_HOG(image)
    glcm_feature = glcm_features(grey, mask=mask)
    colour_feature = extract_color_features(bgr_input, mask=mask)

    combined_feature = np.concatenate([lbp_histogram, hog_feature, glcm_feature, colour_feature])

    return combined_feature
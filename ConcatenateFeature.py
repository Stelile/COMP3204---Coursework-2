import numpy as np
import cv2
from skimage import color
from LocalBinaryPatterns import extract_lbp
from ComputeHistogramGradients import compute_HOG
from glcm import glcm_features
from ColorHistogram import extract_color_features

def extract_feature(image: np.ndarray, mask=None) -> np.ndarray:
    if image.ndim == 3:
        lbp_input = (np.clip(image[:, :, 2], 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb_approx = np.clip(color.hsv2rgb(image) * 255.0, 0, 255).astype(np.uint8)
        bgr_input = cv2.cvtColor(rgb_approx, cv2.COLOR_RGB2BGR)
        grey = cv2.cvtColor(bgr_input, cv2.COLOR_BGR2GRAY)
    else:
        lbp_input = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        grey = lbp_input
        bgr_input = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    _, lbp_histogram = extract_lbp(lbp_input, mask=mask)
    hog_feature = compute_HOG(image)
    glcm_feature = glcm_features(grey, mask=mask)
    colour_result = extract_color_features(bgr_input, mask=mask)
    colour_feature = np.asarray(colour_result["feature_vector"], dtype=np.float64).ravel()
    
    combined = np.concatenate([
        np.asarray(lbp_histogram, dtype=np.float64).ravel(),
        np.asarray(hog_feature, dtype=np.float64).ravel(),
        np.asarray(glcm_feature, dtype=np.float64).ravel(),
        colour_feature,
    ])
    
    if not np.issubdtype(combined.dtype, np.number):
        raise TypeError(f"Non-numeric feature vector produced: {[type(x) for x in combined if not isinstance(x, (int, float, np.number))]}")
    
    return combined
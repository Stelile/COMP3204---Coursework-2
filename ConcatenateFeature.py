import numpy as np
from LocalBinaryPatterns import extract_lbp
from ComputeHistogramGradients import compute_HOG

def extract_feature(image: np.ndarray, mask=None) -> np.ndarray:
    if image.ndim == 3:
        lbp_input = image[:, :, 2]
    else:
        lbp_input = image

    _, lbp_histogram = extract_lbp(lbp_input, mask=mask)
    hog_feature = compute_HOG(image)

    combined_feature = np.concatenate([lbp_histogram, hog_feature])

    return combined_feature
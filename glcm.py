import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2
from ResizeImage import crop_image
from Gaussian import reduce_noise
from leaf_background_removal import remove_leaf_background

DISTANCES = [1]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]   # right, up-right, up, up-left
PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def compute_glcm(grey, levels=16, mask=None):
  
    grey = np.asarray(grey)
    if grey.ndim != 2:
        raise ValueError("not a greyscale image")

    quantised = (grey.astype(np.float64) * levels / 256).astype(np.uint8)

    if mask is None:
        glcm = graycomatrix(quantised, DISTANCES, ANGLES, levels=levels,
                            symmetric=True, normed=True)
        return glcm

    shifted = quantised + 1
    shifted[~mask.astype(bool)] = 0
    glcm = graycomatrix(shifted, DISTANCES, ANGLES, levels=levels + 1,
                        symmetric=True, normed=False)
    glcm = glcm[1:, 1:, :, :].astype(np.float64)

    totals = glcm.sum(axis=(0, 1), keepdims=True)
    totals[totals == 0] = 1
    return glcm / totals


def glcm_features(grey, levels=16, mask=None):
    glcm = compute_glcm(grey, levels=levels, mask=mask)
    features = [graycoprops(glcm, prop).mean() for prop in PROPERTIES]
    return np.array(features)


def extract_glcm(image_path, levels=16):

    rgb = crop_image(image_path)

    rgb = reduce_noise(rgb)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, mask = remove_leaf_background(bgr, return_mask=True)

    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return glcm_features(grey, levels=levels, mask=mask > 0)

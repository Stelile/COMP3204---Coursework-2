"""Extract normalized HSV color features from an isolated plant leaf."""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


ColorFeatures = Dict[str, Union[np.ndarray, float]]


def extract_color_features(
    foreground_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    hue_bins: int = 36,
    hsv_bins: Tuple[int, int, int] = (18, 8, 8),
) -> ColorFeatures:
    """Compute normalized HSV histograms and green/yellow-brown proportions.

    The hue histogram and flattened 3D HSV histogram are each L1-normalized.
    The returned feature vector concatenates both histograms followed by the
    green and yellow-brown proportions.

    Args:
        foreground_bgr: Non-empty, 3-channel, uint8 image in OpenCV BGR order.
        mask: Optional 2D binary foreground mask. Non-zero pixels are included.
            When omitted, non-black image pixels are treated as foreground.
        hue_bins: Number of bins in the 1D hue histogram.
        hsv_bins: Bin counts for the hue, saturation, and value dimensions of
            the 3D HSV histogram.

    Returns:
        A dictionary containing ``hue_histogram``, ``hsv_histogram``,
        ``green_proportion``, ``yellow_brown_proportion``, and ``feature_vector``.
        The two proportions are fractions of all included foreground pixels;
        hues outside the classified ranges remain part of the denominator.

    Raises:
        TypeError: If the image or mask is not a NumPy array, or the image is
            not uint8.
        ValueError: If the image, mask, or histogram bin counts are invalid,
            or if there are no foreground pixels.
    """
    if not isinstance(foreground_bgr, np.ndarray):
        raise TypeError("foreground_bgr must be a NumPy array.")
    if foreground_bgr.ndim != 3 or foreground_bgr.shape[2] != 3:
        raise ValueError("foreground_bgr must have shape (height, width, 3).")
    if foreground_bgr.size == 0:
        raise ValueError("foreground_bgr must not be empty.")
    if foreground_bgr.dtype != np.uint8:
        raise TypeError("foreground_bgr must have dtype uint8.")
    if not isinstance(hue_bins, (int, np.integer)) or hue_bins <= 0:
        raise ValueError("hue_bins must be a positive integer.")
    if (
        len(hsv_bins) != 3
        or any(not isinstance(count, (int, np.integer)) or count <= 0 for count in hsv_bins)
    ):
        raise ValueError("hsv_bins must contain three positive integers.")

    # A provided mask is authoritative; otherwise infer the foreground from
    # the black-background convention used by the background-removal function.
    if mask is None:
        foreground_mask = np.any(foreground_bgr != 0, axis=2)
    else:
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy array.")
        if mask.ndim != 2 or mask.shape != foreground_bgr.shape[:2]:
            raise ValueError("mask must be 2D and match the image height and width.")
        foreground_mask = mask != 0

    pixel_count = int(np.count_nonzero(foreground_mask))
    if pixel_count == 0:
        raise ValueError("No foreground pixels were found.")

    # OpenCV expects an 8-bit single-channel mask, with zero meaning excluded.
    cv_mask = np.ascontiguousarray(foreground_mask.astype(np.uint8) * 255)
    hsv = cv2.cvtColor(foreground_bgr, cv2.COLOR_BGR2HSV)

    # Keep both a compact hue distribution and a joint HSV color distribution.
    hue_histogram = cv2.calcHist(
        [hsv], [0], cv_mask, [int(hue_bins)], [0, 180]
    ).reshape(-1)
    hsv_histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        cv_mask,
        [int(count) for count in hsv_bins],
        [0, 180, 0, 256, 0, 256],
    ).reshape(-1)

    # Normalize each histogram independently so it describes pixel proportions.
    hue_total = float(hue_histogram.sum())
    hsv_total = float(hsv_histogram.sum())
    if hue_total == 0 or hsv_total == 0:
        raise RuntimeError("OpenCV returned an empty histogram for non-empty foreground.")
    hue_histogram = (hue_histogram / hue_total).astype(np.float32)
    hsv_histogram = (hsv_histogram / hsv_total).astype(np.float32)

    hue = hsv[:, :, 0]
    green_count = np.count_nonzero(
        foreground_mask & (hue >= 35) & (hue <= 85)
    )
    yellow_brown_count = np.count_nonzero(
        foreground_mask & (hue >= 5) & (hue <= 34)
    )
    green_proportion = float(green_count / pixel_count)
    yellow_brown_proportion = float(yellow_brown_count / pixel_count)

    # The vector layout is [hue histogram, HSV histogram, green, yellow/brown].
    feature_vector = np.concatenate(
        (
            hue_histogram,
            hsv_histogram,
            np.asarray([green_proportion, yellow_brown_proportion], dtype=np.float32),
        )
    )

    return {
        "hue_histogram": hue_histogram,
        "hsv_histogram": hsv_histogram,
        "green_proportion": green_proportion,
        "yellow_brown_proportion": yellow_brown_proportion,
        "feature_vector": feature_vector,
    }


if __name__ == "__main__":
    foreground = cv2.imread("leaf_foreground.png")
    if foreground is None:
        raise FileNotFoundError(
            "Place an isolated leaf image named 'leaf_foreground.png' in this folder."
        )

    # To use a segmentation mask, load it in grayscale and pass it as mask=mask.
    features = extract_color_features(foreground)
    print(f"Feature vector length: {features['feature_vector'].size}")
    print(f"Green pixels: {features['green_proportion'] * 100:.2f}%")
    print(
        "Yellow/brown pixels: "
        f"{features['yellow_brown_proportion'] * 100:.2f}%"
    )

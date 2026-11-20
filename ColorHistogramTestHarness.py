import unittest

import cv2
import numpy as np

from ColorHistogram import extract_color_features


class TestColorHistogram(unittest.TestCase):
    def test_histograms_and_color_proportions_use_only_masked_pixels(self):
        hsv_pixels = np.array(
            [[[60, 255, 200], [60, 255, 200], [20, 255, 160], [20, 255, 160]]],
            dtype=np.uint8,
        )
        foreground = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR)
        mask = np.array([[255, 255, 255, 0]], dtype=np.uint8)

        features = extract_color_features(
            foreground,
            mask=mask,
            hue_bins=18,
            hsv_bins=(18, 4, 4),
        )

        self.assertAlmostEqual(float(features["hue_histogram"].sum()), 1.0)
        self.assertAlmostEqual(float(features["hsv_histogram"].sum()), 1.0)
        self.assertAlmostEqual(features["green_proportion"], 2 / 3)
        self.assertAlmostEqual(features["yellow_brown_proportion"], 1 / 3)
        self.assertEqual(features["feature_vector"].size, 18 + 18 * 4 * 4 + 2)

    def test_black_background_is_excluded_without_a_mask(self):
        foreground = np.array(
            [[[0, 255, 0], [0, 0, 0], [0, 255, 255]]],
            dtype=np.uint8,
        )

        features = extract_color_features(foreground)

        self.assertAlmostEqual(features["green_proportion"], 0.5)
        self.assertAlmostEqual(features["yellow_brown_proportion"], 0.5)

    def test_empty_foreground_raises_value_error(self):
        foreground = np.zeros((2, 2, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            extract_color_features(foreground)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import cv2
import numpy as np

from ConcatenateFeature import extract_feature
from LocalBinaryPatterns import extract_lbp
from ComputeHistogramGradients import compute_HOG
from glcm import glcm_features
from ColorHistogram import extract_color_features

class TestConcatenateFeature(unittest.TestCase):
    
    #checks the output size against the expected size and format of a 32 by 32 image
    def test_format(self):
        test_image = np.full((32, 32), 100, dtype=np.uint8)
        test_features = extract_feature(test_image)

        #a 32 by 32 image should have LBP = 256, HOG = 324, GLCM = 6, and Colour = 1190 so 1776
        self.assertEqual(test_features.shape, (1776,))
        self.assertTrue(np.issubdtype(test_features.dtype, np.number))
        self.assertTrue(np.isfinite(test_features).all())

    #checks the order of feature vectors in the output
    def test_order(self):
        test_image = np.tile(np.arange(32, dtype=np.uint8) * 8, (32, 1),)

        #should include all pixels
        test_mask = np.full((32, 32), 255, dtype=np.uint8)

        _, expected_lbp = extract_lbp(test_image, mask=test_mask)
        expected_hog = compute_HOG(test_image)
        expected_glcm = glcm_features(test_image, mask=test_mask)

        bgr_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)

        expected_colour = extract_color_features(bgr_image, mask=test_mask)["feature_vector"]
        expected_features = np.concatenate([expected_lbp, expected_hog, expected_glcm, expected_colour,])
        actual_features = extract_feature(test_image, mask=test_mask)

        np.testing.assert_allclose(actual_features, expected_features)

    #checks the usage of a mask by our feature extraction
    def test_handling_mask(self):
        test_image = np.full((32, 32), 200, dtype=np.uint8)
        test_image[8:24, 8:24] = 100
        test_mask = np.zeros((32, 32), dtype=np.uint8)
        test_mask[8:24, 8:24] = 255

        features = extract_feature(test_image, mask=test_mask)

        _, expected_lbp = extract_lbp(test_image, mask=test_mask)
        expected_glcm = glcm_features(test_image, mask=test_mask)

        bgr_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
        expected_colour = extract_color_features(bgr_image, mask=test_mask)["feature_vector"]

        np.testing.assert_allclose(features[:256], expected_lbp)
        np.testing.assert_allclose(features[580:586], expected_glcm)
        np.testing.assert_allclose(features[586:], expected_colour)

if __name__ == "__main__":
    unittest.main(verbosity=2)
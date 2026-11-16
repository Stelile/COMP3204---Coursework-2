import unittest
import numpy as np

from LocalBinaryPatterns import extract_lbp

class TestLBP(unittest.TestCase):

    #when we use this test image our expected code should be 139
    def test_known_pattern(self):
        test_image = np.array([[6, 5, 4], [7, 5, 6], [4, 4, 4]], dtype=np.uint8)
        test_patterns, test_histogram = extract_lbp(test_image)

        np.testing.assert_array_equal(test_patterns, [[139]])

        expected_histogram = np.zeros(256)
        expected_histogram[139] = 1.0

        np.testing.assert_array_equal(test_histogram, expected_histogram)

    def test_mask_neighbourhood_selection(self):
        test_image = np.zeros((7, 7), dtype=np.uint8)
        test_image[2:5, 2:5] = [[6, 5, 4], [7, 5, 6], [4, 4, 4]]

        test_mask = np.zeros((7, 7), dtype=np.uint8)
        test_mask[2:5, 2:5] = 255

        _, test_histogram = extract_lbp(test_image, mask=test_mask)

        expected_histogram = np.zeros(256)
        expected_histogram[139] = 1.0

        np.testing.assert_array_equal(test_histogram, expected_histogram)

if __name__ == "__main__":
    unittest.main(verbosity=2)
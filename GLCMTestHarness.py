import unittest
import numpy as np

from glcm import compute_glcm, glcm_features

class TestGLCM(unittest.TestCase):

    #when we use a constant image there should be no brightness differences
    def test_constant_image(self):
        constant_test_image = np.full((8, 8), 100, dtype=np.uint8)
        test_features = glcm_features(constant_test_image)
        expected_features = np.array([0, 0, 1, 1, 1, 1], dtype=float)

        self.assertEqual(test_features.shape, (6,))
        np.testing.assert_allclose(test_features, expected_features)

    #if we use a colour image the function should reject it
    def test_colour_image(self):
        test_image = np.zeros((8, 8, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            compute_glcm(test_image)

    #directional matrices stored probabilities should sum to one
    def test_proportionality(self):
        test_image = np.tile(np.arange(8, dtype=np.uint8) * 32, (8, 1))
        test_matrix = compute_glcm(test_image)
        test_total = test_matrix.sum(axis=(0, 1))

        #there should be 16 brightness levels, a distance of 1, and 4 directions
        self.assertEqual(test_matrix.shape, (16, 16, 1, 4))

        np.testing.assert_allclose(test_total, np.ones((1, 4)))

    #if we change the pixels outside of the mask our result should be unchanged
    def test_mask_exclusion(self):
        test_image = np.tile(np.arange(8, dtype=np.uint8) * 32, (8, 1))
        test_mask = np.zeros((8, 8), dtype=np.uint8)
        test_mask[2:6, 2:6] = 255

        updated_test_image = test_image.copy()
        updated_test_image[test_mask == 0] = 255

        initial_matrix = compute_glcm(test_image, mask=test_mask)
        updated_matrix = compute_glcm(updated_test_image, mask=test_mask)

        initial_features = glcm_features(test_image, mask=test_mask)
        updated_features = glcm_features(updated_test_image, mask=test_mask)

        np.testing.assert_allclose(initial_matrix, updated_matrix)
        np.testing.assert_allclose(initial_features, updated_features)

if __name__ == "__main__":
    unittest.main(verbosity=2)
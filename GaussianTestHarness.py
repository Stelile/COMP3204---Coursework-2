import unittest
import numpy as np

from Gaussian import reduce_noise

class Test_Gaussian(unittest.TestCase):

    #when we use a sigma of zero the image should remain unchanged but should be stored separately
    def test_zero_sigma(self):
        test_image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        test_result = reduce_noise(test_image, sigma=0)

        np.testing.assert_array_equal(test_result, test_image)

        self.assertFalse(np.shares_memory(test_result, test_image))

    #when we use a constant image it should remain constant
    def test_constant_image(self):
        test_image = np.full((7, 9), 100, dtype=np.uint8)
        test_colour = np.empty((7, 9, 3), dtype=np.uint8)
        test_colour[:] = [30, 100, 200]

        for image in (test_image, test_colour):
            with self.subTest(shape=image.shape):
                test_result = reduce_noise(image)

                self.assertEqual(test_result.shape, image.shape)
                self.assertEqual(test_result.dtype, image.dtype)
            
                np.testing.assert_array_equal(test_result, image)

if __name__ == "__main__":
    unittest.main(verbosity=2)
import os
import glob
import numpy as np
import pytest
from skimage import io, transform, color
from ResizeImage import crop_image

@pytest.mark.parametrize("taret_size",[(224,224),(128,128)])
def test_crop_image_output(plant_village_path, target_size):
    jpg_files = glob.glob(os.path.join(plant_village_path, "**", "*.jpg"), recursive=True)

    sample_files = jpg_files[:5]
    assert len(sample_files) > 0, "No images"

    for img_path in sample_files:
        img_hsv = crop_image(img_path, target_size=target_size)
        expected_shape = (target_size[0], target_size[1], 3)
        assert img_hsv.shape == expected_shape, (f"Expected shape {expected_shape}, got {img_hsv.shape} for {img_path}")
        assert np.issubdtype(img_hsv.dtype, np.floating), (f"Expected floating point, got {img_hsv.dtype} for {img_path}")
        assert np.all(img_hsv >= 0.0) and np.all(img_hsv <= 1.0), (f"HSV values out of range [0.0, 1.0] for {img_path}")

def test_crop_image_dummy():
    dummy_img = np.random.randint(0,256,(300,400,3), dtype=np.uint8)
    dummy_path = "temp_test_image.jpg"
    try:
        io.imsave(dummy_path, dummy_img)
        img_hsv = crop_image(dummy_path, target_size=(224,224))
        assert img_hsv.shape == (224,224,3)
        assert np.all(img_hsv >= 0.0) and np.all(img_hsv <= 1.0)
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
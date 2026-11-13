import numpy as np
import cv2
from skimage import io, transform, color

def reduce_noise(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    if image.ndim not in (2,3):
        raise ValueError("reduce noise expected grey-scale or colour")
    if sigma < 0:
        raise ValueError("reduce noise expected sigma gte zero")
    if sigma == 0:
        return image.copy()

    coordinates = np.arange(-2,3)
    x, y = np.meshgrid(coordinates, coordinates)

    kernel = np.exp(-(x**2 + y**2) / (2 / sigma**2))
    kernel /= kernel.sum()

    border_padding = [(2,2),(2,2)]
    if image.ndim == 3:
        border_padding.append((0,0))

    padded_image = np.pad(image, border_padding, mode="reflect")
    height, width = image.shape[:2]
    smoothed = np.zeros(image.shape, dtype=np.float64)

    for row in range(5):
        for column in range(5):
            neighbours = padded_image[row: row + height, column: column + width]
            smoothed = smoothed + kernel[row, column] * neighbours

    if np.issubdtype(image.dtype, np.integer):
        smoothed = np.rint(smoothed)
    
    return smoothed.astype(image.dtype)

def remove_leaf_background(image, hue_min=20, hue_max=95, sat_min=30, sat_max=255, val_min=20, val_max=255, kernel_size=5, blur_size=5, keep_largest_region=True, black_background=True):
    if image is None:
        raise ValueError("Image is none/could not be read")
    if len(image.shape) != 3:
        raise ValueError("Image must be 3-channel")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_max, sat_max, val_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEc(mask, cv2.MORPH_CLOSE, kernel)

    if blur_size > 1:
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    if keep_largest_region:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

    result = cv2.bitwise_and(image, image, mask=mask)

    if black_background:
        result[mask == 0] = [0, 0 ,0]

    return result

def preprocess_image(image_path, target_size=(224,224), sigma=0.8):
    img = io.imread(image_path)
    h, w = img.shape[:2]
    min_dim = min(h, w)

    start_y = (h - min_dim) // 2
    start_x = (w - min_dim) // 2

    img_cropped = img[start_y + min_dim, start_x: start_x + min_dim]

    sk_target = (target_size[1], target_size[0])
    img_resized = transform.resize(img_cropped, (*sk_target, 3), anti_aliasing=True, preserve_range=True).astype(img_cropped.dtype)

    img_denoised = reduce_noise(img_resized, sigma=sigma)

    if img_denoised.dtype == np.uint8:
        img_bgr = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2BGR)
    else:
        img_uint8 = (img_denoised * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    foreground_bgr = remove_leaf_background(img_bgr)
    foreground_rgb = cv2.cvtColor(foreground_bgr, cv2.COLOR_BGR2RGB)

    if foreground_rgb.dtype == np.uint8:
        img_float = foreground_rgb.astype(np.float32) / 255.0
    else:
        img_float = foreground_rgb

    img_hsv = color.rgb2hsv(img_float)
    return img_hsv
import cv2
import numpy as np


def remove_leaf_background(
    image,
    hue_min=20,
    hue_max=95,
    sat_min=30,
    sat_max=255,
    val_min=20,
    val_max=255,
    kernel_size=5,
    blur_size=5,
    keep_largest_region=True,
    black_background=True,
    return_mask=False,
):
    """
    Remove the background from a plant leaf image by thresholding in HSV color space.

    Parameters:
        image: np.ndarray
            Input image in BGR format (OpenCV default).
        hue_min, hue_max: int
            HSV hue range for leaf-like pixels.
        sat_min, sat_max: int
            HSV saturation range for the leaf mask.
        val_min, val_max: int
            HSV value range controlling brightness.
        kernel_size: int
            Size of the morphological kernel used for clean-up.
        blur_size: int
            Gaussian blur kernel size used to smooth the mask.
        keep_largest_region: bool
            Keep only the largest connected region to remove noise.
        black_background: bool
            Set background pixels to black when True.
        return_mask: bool
            Also return the final binary mask if True.

    Returns:
        result: np.ndarray
            Foreground leaf image with the background removed.
        mask (optional): np.ndarray
            Final binary mask used to isolate the leaf.
    """
    if image is None:
        raise ValueError("Input image is None or could not be read.")

    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-channel BGR image.")

    # Step 1: Convert from BGR to HSV.
    # HSV is commonly preferred for plant segmentation because green and yellow/brown
    # leaf colors are easier to isolate than in RGB, especially against soil or lab tables.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Create a binary mask for the leaf colors.
    # This mask isolates pixels with leaf-like colors and suppresses most background pixels.
    lower = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_max, sat_max, val_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Optional additional mask for diseased/yellow/brown leaf areas.
    # This helps if the plant is partly yellow or brown instead of fully green.
    # lower_brown = np.array([0, 30, 30], dtype=np.uint8)
    # upper_brown = np.array([25, 255, 180], dtype=np.uint8)
    # brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    # mask = cv2.bitwise_or(mask, brown_mask)

    # Step 3: Morphological operations to clean the mask.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Opening removes tiny scattered noise points.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing fills small holes inside the main leaf region.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Smooth the mask using Gaussian blur to reduce jagged edges.
    if blur_size > 1:
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Convert blurred grayscale values back into a binary mask.
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Step 5: Keep only the largest connected component.
    # This removes isolated background fragments that might still remain.
    if keep_largest_region:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

    # Step 6: Apply the mask to the original image.
    result = cv2.bitwise_and(image, image, mask=mask)

    # Step 7: Set the background to black if requested.
    if black_background:
        result[mask == 0] = [0, 0, 0]

    if return_mask:
        return result, mask
    return result


if __name__ == "__main__":
    # Example usage:
    image = cv2.imread("leaf_image.jpg")

    if image is None:
        raise FileNotFoundError("Please place a sample image named 'leaf_image.jpg' in the same folder.")

    # Remove background while preserving the leaf area.
    leaf_foreground = remove_leaf_background(
        image,
        hue_min=20,
        hue_max=95,
        sat_min=30,
        sat_max=255,
        val_min=20,
        val_max=255,
        kernel_size=5,
        blur_size=5,
        keep_largest_region=True,
        black_background=True,
    )

    cv2.imshow("Original", image)
    cv2.imshow("Leaf Only", leaf_foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

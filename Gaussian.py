import numpy as np

#uses a 5 by 5 gaussian filter to reduce the image noise
def reduce_noise(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:

    #checks to see if the image is in a 2/3 dimensional array
    if image.ndim not in (2, 3):
        raise ValueError("Error: the reduce_noise function was expecting either a grey scale or colour image")

    #checks to see if the smoothing strength is appropriate
    if sigma < 0:
        raise ValueError("Error: the reduce_noise function was expecting a sigma value of zero or greater")

    #returns a copy of the original image if the smoothing strength is zero
    if sigma == 0:
        return image.copy()

    #creates the 5 by 5 grid for our filter
    #centered on (0, 0)
    coordinates = np.arange(-2, 3)
    x, y = np.meshgrid(coordinates, coordinates)

    #increases the weight of nearby pixels
    #scales the weights so that they sum to one
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    #used to set 2 pixels of padding along all sides
    #ignores the colour channels if they're present
    border_padding = [(2, 2), (2, 2)]
    if image.ndim == 3:
        border_padding.append((0, 0))

    #used to reflect the pixels at the border
    #also creates an empty array for us to store the result
    padded_image = np.pad(image, border_padding, mode="reflect")
    height, width = image.shape[:2]
    smoothed = np.zeros(image.shape, dtype=np.float64)

    #adds each of the neighbouring weighted pixel values to the output
    for row in range(5):
        for column in range(5):
            neighbours = padded_image[row: row + height, column: column + width]
            smoothed = smoothed + kernel[row, column] * neighbours

    #gives rounded pixel values in case the original image uses integers
    if np.issubdtype(image.dtype, np.integer):
        smoothed = np.rint(smoothed)

    #returns the smoothed image
    return smoothed.astype(image.dtype)
import numpy as np

#this function will return a simple 3 by 3 LBP image and normalised histogram
def extract_lbp(image: np.ndarray, mask=None):

    ##used to check that the image is greyscale and at least 3 by 3 pixels
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("Error: the extract_lbp function was expecting a greyscale image of at least 3 by 3 pixels")

    #makes sure we use only pixels with a full neighbourhood
    centre = image[1:-1, 1:-1]
    height, width = centre.shape

    #create an empty array to store our patterns
    patterns = np.zeros(centre.shape, dtype=np.uint8)

    #used for visiting neighbours clockwise
    offset = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]

    #we initially include every centre pixel in the histogram
    valid = np.ones(centre.shape, dtype=bool)

    #if the function receives a mask use it to identify the leaf pixels
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError("Error: the extract_lbp function received a mask and image with different sizes")

        mask = mask != 0
        valid = mask[1:-1, 1:-1].copy()

    #used to compare each neighbour with its corresponding centre pixel
    for bit, (row, column) in enumerate(offset):
        neighbours = image[row:row + height, column:column + width]

        #we set a neighbour's bit when its value is greater than or equal to its corresponding centre value
        # << bit gives the binary weight to each neighbour
        # |= combines the bits into the final code
        patterns |= (neighbours >= centre).astype(np.uint8) << bit

        #used to make sure we only include pixels fully inside any provided leaf mask
        if mask is not None:
            valid &= mask[row:row + height, column:column + width]
    
    #used to check that there are actually pixels available for the histogram
    if not valid.any():
        raise ValueError("Error: the extract_lbp function received a mask with no usable 3 by 3 neighbourhoods")

    #counts how often each lbp code actually occurs
    histogram = np.bincount(patterns[valid], minlength=256).astype(np.float64)

    #converts the counts into proportions that sum up to one
    histogram = histogram / histogram.sum()

    return patterns, histogram
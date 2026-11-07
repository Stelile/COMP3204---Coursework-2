import numpy as np
# Cite: Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu and the scikit-image contributors. scikit-image: Image processing in Python. PeerJ 2:e453 (2014) https://doi.org/10.7717/peerj.453
from skimage import io, transform, color

def crop_image(image_path, target_size=(224, 224)):
    img = io.imread(image_path)

    h, w = img.shape[:2]
    min_dim = min(h, w)

    start_y = (h-min_dim) // 2
    start_x = (w-min_dim) // 2

    img_cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

    sk_target = (target_size[1], target_size[0])
    img_resized = transform.resize(img_cropped, (*sk_target, 3), anti_aliasing=True, preserve_range=True).astype(img_cropped.dtype)

    # code to convert colour into scikit HSV colour-space
    if img_resized.dtype == np.uint8:
        img_float = img_resized / 255.0
    else:
        img_float = img_resized
        
    img_hsv = color.rgb2hsv(img_float)

    return img_hsv
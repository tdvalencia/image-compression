'''
    Principal Component Analysis (PCA) encoder.
'''

import numpy as np
from skimage import io, color, util
import codec.tools as ct

def apply_pca_prefilter(image_rgb):
    '''
    image_rgb shape: (H, W, 3)
    Returns: rotated_image, rotation_matrix, channel_means
    '''
    H, W, C = image_rgb.shape
    # Flatten to a list of (R, G, B) vectors
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    
    # 1. Center the data (important for PCA)
    means = np.mean(pixels, axis=0)
    centered_pixels = pixels - means
    
    # 2. Compute SVD on the covariance matrix
    # This gives us the optimal rotation for THIS specific image
    cov = np.cov(centered_pixels.T)
    U, S, Vt = np.linalg.svd(cov)
    
    # 3. Rotate pixels into Principal Components
    # PC1 will hold the most energy (Luminance-like)
    pca_pixels = centered_pixels @ Vt.T
    
    rotated_image = pca_pixels.reshape(H, W, 3)
    return rotated_image, Vt, means

def invert_pca(rotated_image, Vt, means):
    H, W, C = rotated_image.shape
    pca_pixels = rotated_image.reshape(-1, 3)
    
    # Invert rotation 
    original_centered = pca_pixels @ Vt
    original_pixels = original_centered + means
    
    clipped_pixels = np.clip(original_pixels, 0, 65535)
    
    return clipped_pixels.reshape(H, W, 3).astype(np.uint16)

if __name__ == '__main__':
    # Load image
    img: np.ndarray = io.imread('theory/deer.ppm')

    # If the image is grayscale, convert it to RGB so our 3-channel math works
    if img.ndim == 2:
        img = color.gray2rgb(img)
    # If the image has an alpha channel (RGBA), strip it to standard RGB
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    # Apply PCA prefilter
    rotated_image, rotation_matrix, channel_means = apply_pca_prefilter(img)

    # For demonstration, let's invert it immediately to check correctness
    inverted_image = invert_pca(rotated_image, rotation_matrix, channel_means)

    # Check if we got back the original (within some numerical tolerance)
    assert np.allclose(img, inverted_image, atol=1), "PCA inversion did not return original image!"
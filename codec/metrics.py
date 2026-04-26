'''
    Will hold evaluations metric functions for the codec.
'''

import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from PIL import Image

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # No error, perfect reconstruction
    max_pixel_value = 1.0  # For images normalized to [0, 1]
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value

def ssim(original, reconstructed):
    # Compute SSIM with proper data range for normalized images
    # Use channel_axis for multichannel images (axis 2 for (H, W, C))
    return ssim_metric(original, reconstructed, data_range=1.0, channel_axis=2, win_size=11)

def compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return float('inf')  # Avoid division by zero
    return original_size / compressed_size

'''
    Will hold evaluations metric functions for the codec.
'''

import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # No error, perfect reconstruction
    max_pixel_value = 65535  # For 16-bit images
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value

def ssim(original, reconstructed):
    # Determine the smaller spatial dimension (H or W)
    min_side = min(original.shape[:2])
    # Set win_size: must be odd, ≤ min_side, and ≤ 11 (default max)
    win_size = min(11, min_side if min_side % 2 == 1 else min_side - 1)
    # Use channel_axis for multichannel images (axis 2 for (H, W, C))
    return ssim_metric(original, reconstructed, data_range=65535, channel_axis=2, win_size=win_size)

def compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return float('inf')  # Avoid division by zero
    return original_size / compressed_size
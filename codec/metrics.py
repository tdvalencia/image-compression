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
    # Determine the smaller spatial dimension (H or W)
    min_side = min(original.shape[:2])
    # Set win_size: must be odd, ≤ min_side, and ≤ 11 (default max)
    win_size = min(11, min_side if min_side % 2 == 1 else min_side - 1)
    # Use channel_axis for multichannel images (axis 2 for (H, W, C))
    return ssim_metric(original, reconstructed, data_range=1.0, channel_axis=2)

def compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return float('inf')  # Avoid division by zero
    return original_size / compressed_size

def comparing_to_jpeg(original, reconstructed):
    # TODO: look at data types and consider if that effects metrics
    jpeg = Image.fromarray((original * 255).astype(np.uint8))
    jpeg.save('temp.jpg', 'JPEG', quality=75)  # Save with a standard quality level
    jpeg_reconstructed = np.array(Image.open('temp.jpg')) / 255.0
    return psnr(original, jpeg_reconstructed), ssim(original, jpeg_reconstructed)
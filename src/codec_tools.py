import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util

# A hardcoded map of exactly how to read an 8x8 block diagonally
# prioritizes the top-left corner (lowest frequency) and ends at the bottom-right (highest frequency)
ZIG_ZAG_INDICES = np.array([
    0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63
])

def zigzag_flatten(block):
    """Flattens an 8x8 block into a 1D array using zig-zag order."""
    # Flatten the block normally first, then reorder it using our map
    return block.flatten()[ZIG_ZAG_INDICES]

def load_and_preprocess_image(filepath: str, block_size: int = 8) -> np.ndarray:
    """Loads an RGB image, converts to float, and crops to fit the block size."""
    img: np.ndarray = io.imread(filepath)

    # If the image is grayscale, convert it to RGB so our 3-channel math works
    if img.ndim == 2:
        img = color.gray2rgb(img)
    # If the image has an alpha channel (RGBA), strip it to standard RGB
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    img = util.img_as_float(img)

    # Crop to ensure dimensions are exactly divisible by the block size
    H, W, C = img.shape
    H_cropped = H - (H % block_size)
    W_cropped = W - (W % block_size)

    return img[:H_cropped, :W_cropped, :]

def block_image(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Breaks the RGB image into blocks"""
    H, W, C = img.shape

    # Reshape into (num_blocks_y, num_blocks_x, channels, block_size, block_size)
    blocks = img.reshape(H // block_size, block_size, W // block_size, block_size, C)
    blocks = blocks.transpose(0, 2, 4, 1, 3)

    return blocks

def plot_zoomed_comparison(original, compressed, title, zoom_size=(1000, 1500)):
    """Plots a zoomed-in center crop of both images side-by-side."""
    H, W, C = original.shape
    zh, zw = zoom_size
    
    # Calculate bounds
    mid_h, mid_w = H // 2, W // 2
    r_start, r_end = max(0, mid_h - zh // 2), min(H, mid_h + zh // 2)
    c_start, c_end = max(0, mid_w - zw // 2), min(W, mid_w + zw // 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the zoomed-in region of the original and compressed images
    axes[0].imshow(original[r_start:r_end, c_start:c_end], vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(compressed[r_start:r_end, c_start:c_end], vmin=0, vmax=1)
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
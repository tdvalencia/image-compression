'''
    Utility functions for loading, preprocessing, saving, and visualizing images in the context of compression.
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
import pickle
import codec.encoders.arithmetic as ae
import codec.encoders.ans as ans
from simple_ans import EncodedSignal

BLOCK_SIZE = 8

def load_and_preprocess_image(filepath: str, block_size: int = BLOCK_SIZE) -> np.ndarray:
    '''Loads an RGB image, converts to 16-bit, and crops to fit the block size.'''
    img: np.ndarray = io.imread(filepath)

    # If the image is grayscale, convert it to RGB so our 3-channel math works
    if img.ndim == 2:
        img = color.gray2rgb(img)
    # If the image has an alpha channel (RGBA), strip it to standard RGB
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    img = util.img_as_uint(img) # consider switching to 8-bit: img_as_ubyte(img)

    # Crop to ensure dimensions are exactly divisible by the block size
    H, W, C = img.shape
    H_cropped = H - (H % block_size)
    W_cropped = W - (W % block_size)

    return img[:H_cropped, :W_cropped, :]

def block_image(img: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    '''Breaks the RGB image into blocks'''
    H, W, C = img.shape

    # Reshape into (num_blocks_y, num_blocks_x, channels, block_size, block_size)
    blocks = img.reshape(H // block_size, block_size, W // block_size, block_size, C)
    blocks = blocks.transpose(0, 2, 4, 1, 3)

    return blocks

def unblock_image(blocks: np.ndarray, image_shape: tuple) -> np.ndarray:
    '''Reconstructs the image from blocks and removes any padding'''
    num_blocks_y, num_blocks_x, C, block_size, _ = blocks.shape
    H, W, _ = image_shape

    # 1. Reverse the blocking process
    blocks = blocks.transpose(0, 3, 1, 4, 2)
    
    # 2. Reshape to the FULL, padded dimensions (e.g., 808x808)
    padded_height = num_blocks_y * block_size
    padded_width = num_blocks_x * block_size
    img_padded = blocks.reshape(padded_height, padded_width, C)

    # 3. Crop off the extra padding to restore the original size (e.g., 803x805)
    img_cropped = img_padded[:H, :W, :]

    return img_cropped

def build_quantization_matrix(block_size=8, quality_scaler=1.0):
    q_matrix = np.zeros((block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            # Quadratic growth: 
            # Low frequencies stay in the 5,000s, high frequencies explode past 200,000
            q_matrix[i, j] = 5000 + ((i + j)**2) * 8000 * quality_scaler
            
    # Protect the DC coefficient to preserve overall block brightness
    q_matrix[0, 0] = 2000 * quality_scaler 
    return q_matrix

def save_uofm_container(filename, shape, metadata, bitstreams):
    '''
    A basic container that packs the minimal required payload into bytes.
    All encoder-specific formatting should be handled before calling this.
    '''
    payload = {
        'shape': shape,
        'metadata': metadata,
        'bitstreams': bitstreams
    }

    with open(filename, 'wb') as file:
        pickle.dump(payload, file)

def load_uofm_container(filename):
    '''
    Loads the pickled bytes and returns the structural components.
    The calling script is responsible for unpacking the bitstreams and metadata.
    '''
    with open(filename, 'rb') as file:
        payload = pickle.load(file)
        
    shape = payload['shape']
    metadata = payload['metadata']
    bitstreams = payload['bitstreams']
    
    return shape, metadata, bitstreams

def plot_zoomed_comparison(original, compressed, title, zoom_size=(1000, 1500)):
    '''Plots a zoomed-in center crop of both images side-by-side.'''
    H, W, C = original.shape
    zh, zw = zoom_size
    
    # Calculate bounds
    mid_h, mid_w = H // 2, W // 2
    r_start, r_end = max(0, mid_h - zh // 2), min(H, mid_h + zh // 2)
    c_start, c_end = max(0, mid_w - zw // 2), min(W, mid_w + zw // 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the zoomed-in region of the original and compressed images
    axes[0].imshow(original[r_start:r_end, c_start:c_end], vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(compressed[r_start:r_end, c_start:c_end], vmin=0, vmax=1)
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
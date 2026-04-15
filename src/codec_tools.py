'''
    Utility functions for loading, preprocessing, saving, and visualizing images in the context of compression.
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
import pickle
import src.encoders.arithmetic as ae
import src.encoders.run_length as rle

def load_and_preprocess_image(filepath: str, block_size: int = 8) -> np.ndarray:
    '''Loads an RGB image, converts to float, and crops to fit the block size.'''
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

def save_uofm_container(filename, original_shape, encoder_type, encoder_metadata, compressed_bits):
    '''
    A universal container that saves compressed data regardless of the algorithm used.
    '''
    print(f'Packaging {encoder_type.upper()} data for {filename}...')
    
    # 1. Pack the bits into bytes
    bit_string = ''.join(str(b) for b in compressed_bits)
    padding_length = (8 - (len(bit_string) % 8)) % 8
    bit_string += '0' * padding_length
    byte_array = bytearray(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

    # 2. Build the Universal Container Payload
    payload = {
        'signature': 'UOFM_CONTAINER_V2', 
        'shape': original_shape,      
        'encoder_type': encoder_type,         # e.g., 'huffman', 'ans', 'arithmetic'
        'encoder_metadata': encoder_metadata, # flexible dictionary for any extra info the encoder needs to decode (e.g., probability tables)
        'padding': padding_length,    
        'data': byte_array            
    }

    # 3. Save it
    with open(filename, 'wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_uofm_container(filename):
    with open(filename, 'rb') as file:
        payload = pickle.load(file)
        
    if payload.get('signature') != 'UOFM_CONTAINER_V2':
        raise ValueError("Unrecognized file format!")
        
    # 1. Unpack the universal data
    shape = payload['shape']
    encoder_type = payload['encoder_type']
    metadata = payload['encoder_metadata']
    
    # 2. Unpack the bytes back into bits
    raw_bytes = payload['data']
    bit_string = "".join(f"{byte:08b}" for byte in raw_bytes)
    if payload['padding'] > 0:
        bit_string = bit_string[:-payload['padding']]
    bit_list = [int(b) for b in bit_string]
    
    # 3. Route the data to the correct Decoder based on the flag!
    if encoder_type == 'huffman':
        print("Detected Huffman encoding. Booting Huffman decoder...")
        tree = metadata['huffman_tree']
        # decoded_rle = run_huffman_decoder(bit_list, tree)
        
    elif encoder_type in ['arithmetic', 'ans']:
        print(f"Detected {encoder_type.upper()} encoding. Booting fractional decoder...")
        probs = metadata['probabilities']
        totals = metadata['total_symbols']
        decoded_rle = ae.decode_rle(bit_list, probs, totals)
        
    else:
        raise ValueError(f"I don't know how to decode: {encoder_type}")
        
    # Return the decoded RLE array and the shape so the rest of your pipeline 
    # (Inverse ZigZag -> IDCT) can finish the job!
    return shape, decoded_rle

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
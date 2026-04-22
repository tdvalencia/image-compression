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

def load_and_preprocess_image(filepath: str, block_size: int = 8) -> np.ndarray:
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

def save_uofm_container(filename, shape, encoder_type, encoder_metadata, compressed_data):
    '''
    Universal container that handles both dahuffman (bytes) and Arithmetic (bit list).
    '''
    # 1. If the data is already packed into bytes (like from dahuffman or ANS)
    if isinstance(compressed_data, bytes):
        # Pad bytes to make length divisible by 4 (for uint32)
        remainder = len(compressed_data) % 4
        padding = (4 - remainder) % 4  # 0 if already divisible, otherwise 1-3
        raw_bytes = compressed_data + (b'\x00' * padding)
    else:
        # 2. If it's a list of bits from Arithmetic Encoding, we manually pack it
        if isinstance(compressed_data, list):
            compressed_data = "".join(str(b) for b in compressed_data)
            
        padding = 8 - (len(compressed_data) % 8)
        if padding == 8: 
            padding = 0
            
        padded_bit_string = compressed_data + ("0" * padding)
        raw_bytes = bytearray(int(padded_bit_string[i:i+8], 2) for i in range(0, len(padded_bit_string), 8))

    # 3. Assemble and save the payload
    payload = {
        'signature': 'UOFM_CONTAINER_V2',
        'shape': shape,
        'padding': padding,
        'encoder_type': encoder_type,
        'encoder_metadata': encoder_metadata,
        'data': raw_bytes
    }

    with open(filename, 'wb') as file:
        pickle.dump(payload, file)

def load_uofm_container(filename):
    with open(filename, 'rb') as file:
        payload = pickle.load(file)
        
    shape = payload['shape']
    encoder_type = payload['encoder_type']
    metadata = payload['encoder_metadata']
    raw_bytes = payload['data']
    
    if encoder_type == 'huffman':
        # Parse the strings ('val_count') back into RLE tuples ((val, count))
        from codec.encoders import huffman as hf
        decoded_rle = hf.decode_rle(raw_bytes, metadata['symbol_counts'])
            
    elif encoder_type == 'arithmetic':
        # Unpack the bytes back to bits just like you normally do for AE
        bit_string = "".join(f"{byte:08b}" for byte in raw_bytes)
        if payload['padding'] > 0:
            bit_string = bit_string[:-payload['padding']]
        bit_list = [int(b) for b in bit_string]
        
        symbol_counts = metadata['symbol_counts']
        totals = metadata['total_symbols']
        decoded_rle = ae.decode_rle(bit_list, symbol_counts, totals)

    elif encoder_type == 'ans':
        print(f"Raw bytes length from pickle: {len(raw_bytes)}")
        print(f"Padding from payload: {payload['padding']}")
        print(f"Bytes length from metadata: {metadata['bytes_length']}")
        
        # Use the exact bytes_length stored in metadata
        bytes_length = metadata['bytes_length']
        actual_bytes = raw_bytes[:bytes_length]
        
        print(f"Actual bytes length after extraction: {len(actual_bytes)}, divisible by 4: {len(actual_bytes) % 4 == 0}")
        
        words_length = metadata['words_length']
        # Use uint32, not uint64!
        words = np.frombuffer(actual_bytes, dtype=np.uint32)[:words_length]
        
        encoded_signal = EncodedSignal(
            state=np.uint64(metadata['state']),
            symbol_counts=metadata['counts'],
            symbol_values=metadata['values'],
            signal_length=metadata['length'],
            words=words
        )
        
        decoded_rle = ans.decode_rle(encoded_signal)

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
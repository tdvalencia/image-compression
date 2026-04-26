from itertools import groupby
import numpy as np

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

INVERSE_ZIG_ZAG_INDICES = np.argsort(ZIG_ZAG_INDICES)

def zigzag_flatten(block):
    """Flattens an 8x8 block into a 1D array using zig-zag order."""
    # Flatten the block normally first, then reorder it using our map
    return block.flatten()[ZIG_ZAG_INDICES]

def zigzag_unflatten(flat_array):
    """
    Takes a 1D array of 64 zig-zagged elements 
    and reconstructs the original 8x8 matrix.
    """
    # Step A: Un-shuffle the 1D array back into standard "book reading" order
    unshuffled_flat = np.array(flat_array)[INVERSE_ZIG_ZAG_INDICES]
    
    # Step B: Fold that flat list back into an 8x8 grid
    reconstructed_block = unshuffled_flat.reshape((8, 8))
    
    return reconstructed_block

def encode(data):
    '''
    A highly compact, generalized RLE function.
    Works on strings, lists, tuples, or any iterable.
    '''
    # groupby groups consecutive identical elements. 
    # We turn the group into a list to get its length.
    return [(key, len(list(group))) for key, group in groupby(data)]

def decode(rle_data):
    '''
    Inverse of run_length_encode.
    Expects an iterable of (value, count) pairs and expands them.
    '''
    out = []
    for val, count in rle_data:
        out.extend([val] * int(count))
    return out

def decode_master_rle_list(rle_tuples):
    '''
    Decodes a continuous, flattened list of RLE tuples for an entire image.
    Safely pads EOBs across multiple 64-item boundaries.
    '''
    flat_frequencies = []
    
    for val, count in rle_tuples:
        if count == 'EOB':
            # Calculate how many numbers are currently in our flat list
            current_length = len(flat_frequencies)
            
            # Find out how far we are into the current 8x8 block
            remainder = current_length % 64
            
            # Pad the exact number of zeros needed to reach the next multiple of 64
            padding_needed = 64 - remainder
            flat_frequencies.extend([0] * padding_needed)
        else:
            # Standard RLE expansion
            flat_frequencies.extend([val] * int(count))
            
    return flat_frequencies
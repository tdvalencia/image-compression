import src.codec_tools as ct
import src.encoders.run_length as rle
import numpy as np
from scipy.fft import idctn

BLOCK_SIZE = 8

if __name__ == '__main__':
    # 1. Load the metadata and the RLE tuples from the custom file
    # shape is the original image tuple, e.g., (Height, Width, Channels)
    shape, decoded_rle_tuples = ct.load_uofm_container('deer_compressed.uofm')
    H, W, C = shape
    
    # Calculate how many blocks we should expect
    num_blocks_y = H // BLOCK_SIZE
    num_blocks_x = W // BLOCK_SIZE

    # 2. RLE DECODE
    # Expand [(1, 145), (63, 0)] back out into [145, 0, 0, 0...]
    # (Assuming you have an rle.decode function to reverse rle.encode)
    flat_frequencies = rle.decode_master_rle_list(decoded_rle_tuples)

    print(f"Total frequencies decoded: {len(flat_frequencies)}")

    # 3. INVERSE ZIG-ZAG
    # Group the massive flat list into chunks of 64, then reverse the zig-zag
    total_blocks = len(flat_frequencies) // 64
    chunked_freqs = np.array(flat_frequencies).reshape((total_blocks, 64))
    
    reconstructed_8x8_blocks = np.array([
        rle.zigzag_unflatten(chunk) for chunk in chunked_freqs
    ])

    # 4. RESHAPE TO 5D TENSOR
    # Put the blocks back into the strict layout the IDCT requires
    blocked_frequencies = reconstructed_8x8_blocks.reshape(
        (num_blocks_y, num_blocks_x, C, BLOCK_SIZE, BLOCK_SIZE)
    )

    # 5. INVERSE DCT
    # This is the magic! Convert the math back into colors.
    # norm='ortho' must match what you used in the encoder exactly
    raw_pixel_blocks = np.asarray(idctn(blocked_frequencies, axes=(3, 4), norm='ortho'))

    # 6. UNBLOCK THE IMAGE
    reconstructed_image_floats = ct.unblock_image(raw_pixel_blocks, image_shape=shape)

    # 7. LOAD ORIGINAL IMAGE EARLY (We need it to check the scale!)
    original_image = ct.load_and_preprocess_image('images/rgb16bit/deer.ppm', block_size=BLOCK_SIZE)
    max_val = original_image.max()

    # 8. THE DYNAMIC FINAL POLISH
    if max_val <= 1.0:
        # It's a normalized float image (0.0 to 1.0)
        final_image = np.clip(reconstructed_image_floats, 0.0, 1.0)
    elif max_val > 255:
        # It's a 16-bit integer image (0 to 65535)
        final_image = np.clip(np.round(reconstructed_image_floats), 0, 65535).astype(np.uint16)
    else:
        # It's a standard 8-bit image (0 to 255)
        final_image = np.clip(np.round(reconstructed_image_floats), 0, 255).astype(np.uint8)

    # 9. VIEW RESULTS
    ct.plot_zoomed_comparison(original_image, final_image, title='Reconstructed Compressed Image')
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

    # 5. DEQUANTIZATION
    # Multiply by the same quantization factor used in compression to restore scale
    q_matrix = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            q_matrix[i, j] = 1000 + (i + j) * 4000
    q_matrix[0, 0] = 8000
    restored_frequencies = blocked_frequencies * q_matrix

    # 6. INVERSE DCT
    # norm='ortho' must match what you used in the encoder exactly
    # round and clip to valid pixel range after IDCT
    raw_floats = np.asarray(idctn(restored_frequencies, axes=(3, 4), norm='ortho'))
    raw_pixel_blocks = np.clip(np.round(raw_floats), 0, 65535).astype(np.uint16)

    # 7. UNBLOCK THE IMAGE
    reconstructed_image = ct.unblock_image(raw_pixel_blocks, image_shape=shape)
    final_image = np.clip(reconstructed_image, 0, 65535).astype(np.uint16)

    print(f"Max reconstructed pixel: {reconstructed_image.max()}")

    # 8. LOAD ORIGINAL IMAGE EARLY (We need it to check the scale!)
    original_image = ct.load_and_preprocess_image('images/rgb16bit/deer.ppm', block_size=BLOCK_SIZE)
    original_image = original_image[:H, :W, :]

    # 10. VIEW RESULTS
    # Matplotlib cannot render uint16. We must convert the arrays to 0.0 - 1.0 floats strictly for the plotter!
    display_original = original_image.astype(np.float32) / 65535.0
    display_reconstructed = final_image.astype(np.float32) / 65535.0
    ct.plot_zoomed_comparison(display_original, display_reconstructed, title='Reconstructed Compressed Image')
import codec.tools as ct
import codec.encoders.run_length as rle
import numpy as np
from scipy.fft import idctn
import codec.metrics as metrics
import os

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

    # Matplotlib and skimage.metrics dont like uint16. We must convert the arrays to 0.0 - 1.0 floats
    original_float_norm = original_image.astype(np.float32) / 65535.0
    reconstructed_float_norm = final_image.astype(np.float32) / 65535.0

    # 9. EVALUATE METRICS AND VISUALIZE
    psnr_value = metrics.psnr(original_float_norm, reconstructed_float_norm)
    ssim_value = metrics.ssim(original_float_norm, reconstructed_float_norm)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

    # compression ratio
    original_size = H * W * C * 2  # 2 bytes per pixel in 16-bit RGB
    compressed_size = os.path.getsize('deer_compressed.uofm')
    compression_ratio = original_size / compressed_size
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}:1")

    # 10. VIEW RESULTS
    ct.plot_zoomed_comparison(original_float_norm, reconstructed_float_norm, title='Reconstructed Compressed Image')
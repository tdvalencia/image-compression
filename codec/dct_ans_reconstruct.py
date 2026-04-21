import numpy as np
import codec.tools as ct
import codec.encoders.ans as ans
from simple_ans import EncodedSignal
import codec.encoders.run_length as rle
from scipy.fft import idctn
import codec.metrics as metrics
import os

def build_quantization_matrix(block_size=8):
    q_matrix = np.zeros((block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            q_matrix[i, j] = 100 + (i + j) * 200  # Much less aggressive
    q_matrix[0, 0] = 50
    return q_matrix

def reconstruct_from_ans(input_file='deer_ans_compressed.uofm', original_path='images/rgb16bit/deer.ppm'):
    shape, decoded_rle = ct.load_uofm_container(input_file)
    H, W, C = shape

    # Don't call rle.decode() - decoded_rle is already RLE tuples
    print(f'Total RLE tuples decoded: {len(decoded_rle)}')

    flat_frequencies = rle.decode_master_rle_list(decoded_rle)  # Pass decoded_rle directly
    print(f'Total frequencies decoded: {len(flat_frequencies)}')

    total_blocks = len(flat_frequencies) // 64
    chunked_freqs = np.array(flat_frequencies).reshape((total_blocks, 64))
    reconstructed_8x8_blocks = np.array([rle.zigzag_unflatten(chunk) for chunk in chunked_freqs])

    num_blocks_y = H // 8
    num_blocks_x = W // 8
    blocked_frequencies = reconstructed_8x8_blocks.reshape(
        (num_blocks_y, num_blocks_x, C, 8, 8)
    )

    q_matrix = build_quantization_matrix(8)
    restored_frequencies = blocked_frequencies * q_matrix

    raw_floats = np.asarray(idctn(restored_frequencies, axes=(3, 4), norm='ortho'))
    raw_pixel_blocks = np.clip(np.round(raw_floats), 0, 65535).astype(np.uint16)

    reconstructed_image = ct.unblock_image(raw_pixel_blocks, image_shape=shape)
    final_image = np.clip(reconstructed_image, 0, 65535).astype(np.uint16)

    original_image = ct.load_and_preprocess_image(original_path, block_size=8)
    original_image = original_image[:H, :W, :]

    print(f'Original image shape: {original_image.shape}, dtype: {original_image.dtype}')
    print(f'Original image range: [{original_image.min()}, {original_image.max()}]')
    print(f'Reconstructed image shape: {final_image.shape}, dtype: {final_image.dtype}')
    print(f'Reconstructed image range: [{final_image.min()}, {final_image.max()}]')

    original_float_norm = original_image.astype(np.float32) / 65535.0
    reconstructed_float_norm = final_image.astype(np.float32) / 65535.0

    print(f'Normalized original range: [{original_float_norm.min():.4f}, {original_float_norm.max():.4f}]')
    print(f'Normalized reconstructed range: [{reconstructed_float_norm.min():.4f}, {reconstructed_float_norm.max():.4f}]')

    psnr_value = metrics.psnr(original_float_norm, reconstructed_float_norm)
    ssim_value = metrics.ssim(original_float_norm, reconstructed_float_norm)

    print(f'Max reconstructed pixel: {final_image.max()}')
    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'SSIM: {ssim_value:.4f}')

    ct.plot_zoomed_comparison(original_float_norm, reconstructed_float_norm, title=f'Zoomed Comparison (PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f})')

if __name__ == '__main__':
    reconstruct_from_ans()
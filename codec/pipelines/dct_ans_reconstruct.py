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
            # Quadratic scaling: Low frequencies stay below 10,000, 
            # High frequencies skyrocket into the 100,000s
            q_matrix[i, j] = 1000 + ((i + j)**2) * 2500
    q_matrix[0, 0] = 500
    return q_matrix

def reconstruct_from_ans(input_file='deer_ans_compressed.uofm', original_path='images/rgb16bit/deer.ppm'):
    shape, metadata, bitstreams = ct.load_uofm_container(input_file)
    H, W, C = shape

    words = np.frombuffer(
        bitstreams['ans_words'],
        dtype=np.uint32,
        count=metadata['words_length'],
    ).copy()
    encoded_signal = EncodedSignal(
        np.uint64(metadata['state']),
        words,
        metadata['counts'],
        metadata['values'],
        int(metadata['length']),
    )
    decoded_rle = ans.decode_rle(encoded_signal)

    flat_frequencies = rle.decode_master_rle_list(decoded_rle)
    # print(f'Total frequencies decoded: {len(flat_frequencies)}')

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

    reconstructed_image = ct.unblock_image(raw_pixel_blocks, shape)
    final_image = np.clip(reconstructed_image, 0, 65535).astype(np.uint16)

    original_image = ct.load_and_preprocess_image(original_path, block_size=8)
    original_image = original_image[:H, :W, :]

    # print(f'Original image shape: {original_image.shape}, dtype: {original_image.dtype}')
    # print(f'Original image range: [{original_image.min()}, {original_image.max()}]')
    # print(f'Reconstructed image shape: {final_image.shape}, dtype: {final_image.dtype}')
    # print(f'Reconstructed image range: [{final_image.min()}, {final_image.max()}]')

    original_float_norm = original_image.astype(np.float32) / 65535.0
    reconstructed_float_norm = final_image.astype(np.float32) / 65535.0

    # print(f'Normalized original range: [{original_float_norm.min():.4f}, {original_float_norm.max():.4f}]')
    # print(f'Normalized reconstructed range: [{reconstructed_float_norm.min():.4f}, {reconstructed_float_norm.max():.4f}]')

    psnr_value = metrics.psnr(original_float_norm, reconstructed_float_norm)
    ssim_value = metrics.ssim(original_float_norm, reconstructed_float_norm)

    print(f'Max reconstructed pixel: {final_image.max()}')
    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'SSIM: {ssim_value:.4f}')

    # compression ratio
    original_size = H * W * C * 2  # 2 bytes per pixel in 16-bit RGB
    compressed_size = os.path.getsize(input_file)
    compression_ratio = original_size / compressed_size
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}:1")

    ct.plot_zoomed_comparison(original_float_norm, reconstructed_float_norm, title=f'Zoomed Comparison (PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f})')

if __name__ == '__main__':
    reconstruct_from_ans(input_file='fireworks_ans_compressed.uofm', original_path='images/rgb16bit/fireworks.ppm')
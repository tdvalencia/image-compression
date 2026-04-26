import codec.tools as ct
import codec.encoders.run_length as rle
import codec.encoders.arithmetic as ae
import codec.metrics as metrics
import numpy as np
from scipy.fft import idctn

BLOCK_SIZE = 8
INPUT_FILE = 'deer_compressed_arithmetic.uofm'
ORIGINAL_IMAGE = 'images/rgb16bit/deer.ppm'


def build_quantization_matrix(block_size=BLOCK_SIZE):
    q_matrix = np.zeros((block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            q_matrix[i, j] = 1000 + (i + j) * 4000
    q_matrix[0, 0] = 8000
    return q_matrix


def reconstruct_from_arithmetic(input_file=INPUT_FILE, original_path=ORIGINAL_IMAGE):
    shape, metadata, bitstreams = ct.load_uofm_container(input_file)
    H, W, C = shape

    decoded_rle_tuples = ae.decode_rle(
        bitstreams['ae_bits'],
        metadata['symbol_counts'],
        metadata['total_symbols'],
    )

    num_blocks_y = H // BLOCK_SIZE
    num_blocks_x = W // BLOCK_SIZE

    flat_frequencies = rle.decode_master_rle_list(decoded_rle_tuples)
    print(f'Total frequencies decoded: {len(flat_frequencies)}')

    # debugging
    print(f"Shape: {shape}")
    expected_len = num_blocks_y * num_blocks_x * C * 64
    print(f"Expected flat_frequencies length: {expected_len}")
    print(f"Actual flat_frequencies length: {len(flat_frequencies)}")

    total_blocks = len(flat_frequencies) // 64
    chunked_freqs = np.array(flat_frequencies).reshape((total_blocks, 64))
    reconstructed_8x8_blocks = np.array([rle.zigzag_unflatten(chunk) for chunk in chunked_freqs])

    blocked_frequencies = reconstructed_8x8_blocks.reshape(
        (num_blocks_y, num_blocks_x, C, BLOCK_SIZE, BLOCK_SIZE)
    )

    q_matrix = build_quantization_matrix(BLOCK_SIZE)
    restored_frequencies = blocked_frequencies * q_matrix

    raw_floats = np.asarray(idctn(restored_frequencies, axes=(3, 4), norm='ortho'))
    raw_pixel_blocks = np.clip(np.round(raw_floats), 0, 65535).astype(np.uint16)

    reconstructed_image = ct.unblock_image(raw_pixel_blocks, shape)
    final_image = np.clip(reconstructed_image, 0, 65535).astype(np.uint16)

    original_image = ct.load_and_preprocess_image(original_path, block_size=BLOCK_SIZE)
    original_image = original_image[:H, :W, :]

    original_float_norm = original_image.astype(np.float32) / 65535.0
    reconstructed_float_norm = final_image.astype(np.float32) / 65535.0

    psnr_value = metrics.psnr(original_float_norm, reconstructed_float_norm)
    ssim_value = metrics.ssim(original_float_norm, reconstructed_float_norm)

    print(f'Max reconstructed pixel: {final_image.max()}')
    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'SSIM: {ssim_value:.4f}')

    ct.plot_zoomed_comparison(original_float_norm, reconstructed_float_norm,
                              title='Reconstructed Arithmetic Compressed Image')

    return final_image


if __name__ == '__main__':
    reconstruct_from_arithmetic()

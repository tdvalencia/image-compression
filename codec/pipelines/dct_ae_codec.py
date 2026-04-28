'''
    DCT, quantization, masking,and Arithmetic coding.
'''

import numpy as np
from scipy.fft import dctn, idctn

import codec.tools as ct
import codec.encoders.run_length as rle
import codec.encoders.arithmetic as ae

BLOCK_SIZE = 8
MASK_SIZE = 2


def build_mask(block_size: int = BLOCK_SIZE, mask_size: int = MASK_SIZE) -> np.ndarray:
    mask = np.zeros((block_size, block_size), dtype=np.int32)
    mask[:mask_size, :mask_size] = 1
    return mask

def compress_image(image_path: str, output_path: str) -> None:
    image = ct.load_and_preprocess_image(image_path, block_size=BLOCK_SIZE)
    blocked_image = ct.block_image(image, block_size=BLOCK_SIZE)
    dct_blocks = np.asarray(dctn(blocked_image, axes=(3, 4), norm='ortho'))

    q_matrix = ct.build_quantization_matrix(BLOCK_SIZE)
    quantized_blocks = np.round(dct_blocks / q_matrix).astype(np.int32)

    mask = build_mask(BLOCK_SIZE, MASK_SIZE)
    masked_blocks = quantized_blocks * mask
    block_matrix = masked_blocks.reshape(-1, BLOCK_SIZE, BLOCK_SIZE).astype(np.int32)

    rle_blocks: list = []
    for block in block_matrix:
        flat_zig_zag = rle.zigzag_flatten(block)
        rle_blocks.extend(rle.encode(flat_zig_zag))

    compressed_bits, symbol_counts, total_symbols = ae.encode_rle(rle_blocks)
    metadata = {
        'symbol_counts': symbol_counts,
        'total_symbols': total_symbols,
    }
    bitstreams = {'ae_bits': compressed_bits}
    ct.save_uofm_container(output_path, image.shape, metadata, bitstreams)


def reconstruct_image(compressed_path: str) -> np.ndarray:
    shape, metadata, bitstreams = ct.load_uofm_container(compressed_path)
    H, W, C = shape

    decoded_rle_tuples = ae.decode_rle(
        bitstreams['ae_bits'],
        metadata['symbol_counts'],
        metadata['total_symbols'],
    )

    num_blocks_y = H // BLOCK_SIZE
    num_blocks_x = W // BLOCK_SIZE

    flat_frequencies = rle.decode_master_rle_list(decoded_rle_tuples)
    total_blocks = len(flat_frequencies) // 64
    chunked_freqs = np.array(flat_frequencies).reshape((total_blocks, 64))
    reconstructed_8x8_blocks = np.array(
        [rle.zigzag_unflatten(chunk) for chunk in chunked_freqs]
    )

    blocked_frequencies = reconstructed_8x8_blocks.reshape(
        (num_blocks_y, num_blocks_x, C, BLOCK_SIZE, BLOCK_SIZE)
    )

    q_matrix = ct.build_quantization_matrix(BLOCK_SIZE)
    restored_frequencies = blocked_frequencies.astype(np.float32) * q_matrix

    raw_floats = np.asarray(idctn(restored_frequencies, axes=(3, 4), norm='ortho'))
    raw_pixel_blocks = np.clip(np.round(raw_floats), 0, 65535).astype(np.uint16)
    reconstructed_image = ct.unblock_image(raw_pixel_blocks, shape)
    return np.clip(reconstructed_image, 0, 65535).astype(np.uint16)


if __name__ == '__main__':
    import os
    import codec.metrics as metrics
    import codec.tools as ct

    inp = 'images/rgb16bit/deer.ppm'
    out = 'deer_compressed.uofm'
    compress_image(inp, out)
    rec = reconstruct_image(out)

    # Evaluate metrics
    img = ct.load_and_preprocess_image(inp)

    print('Computing metrics...')
    float_original = img.astype(np.float32) / 65535.0
    float_reconstructed = rec.astype(np.float32) / 65535.0

    psnr_value = metrics.psnr(float_original, float_reconstructed)
    ssim_value = metrics.ssim(float_original, float_reconstructed)

    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'SSIM: {ssim_value:.4f}')

    original_size = img.size * img.dtype.itemsize
    compressed_size = os.path.getsize(out)
    compression_ratio = original_size / compressed_size

    print(f'Input image shape: {img.shape}')
    print(f'Original uncompressed bytes: {original_size:,}')
    print(f'Compressed file bytes: {compressed_size:,}')
    print(f'Compression ratio: {compression_ratio:.2f}:1')

    ct.plot_zoomed_comparison(float_original, float_reconstructed, title="DCT AE Compression")

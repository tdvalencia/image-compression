import os
import numpy as np
from scipy.fft import dctn
import codec.tools as ct
import codec.encoders.run_length as rle
import codec.encoders.arithmetic as ae

BLOCK_SIZE = 8
MASK_SIZE = 2
INPUT_IMAGE = 'images/rgb16bit/deer.ppm'
OUTPUT_FILE = 'deer_compressed_arithmetic.uofm'


def build_quantization_matrix(block_size=BLOCK_SIZE):
    q_matrix = np.zeros((block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            q_matrix[i, j] = 1000 + (i + j) * 4000
    q_matrix[0, 0] = 8000
    return q_matrix


def build_mask(block_size=BLOCK_SIZE, mask_size=MASK_SIZE):
    mask = np.zeros((block_size, block_size), dtype=np.int32)
    mask[:mask_size, :mask_size] = 1
    return mask


def compress_with_dct_and_arithmetic(input_path=INPUT_IMAGE, output_path=OUTPUT_FILE):
    image = ct.load_and_preprocess_image(input_path, block_size=BLOCK_SIZE)
    blocked_image = ct.block_image(image, block_size=BLOCK_SIZE)

    dct_blocks = np.asarray(dctn(blocked_image, axes=(3, 4), norm='ortho'))

    q_matrix = build_quantization_matrix(BLOCK_SIZE)
    quantized_blocks = np.round(dct_blocks / q_matrix).astype(np.int32)

    mask = build_mask(BLOCK_SIZE, MASK_SIZE)
    masked_blocks = quantized_blocks * mask

    block_matrix = masked_blocks.reshape(-1, BLOCK_SIZE, BLOCK_SIZE).astype(np.int32)

    rle_blocks = []
    for block in block_matrix:
        flat_zig_zag = rle.zigzag_flatten(block)
        rle_blocks.extend(rle.encode(flat_zig_zag))

    compressed_bits, symbol_counts, total_symbols = ae.encode_rle(rle_blocks)
    metadata = {
        'symbol_counts': symbol_counts,
        'total_symbols': total_symbols
    }

    bitstreams = {'ae_bits': compressed_bits}
    ct.save_uofm_container(output_path, image.shape, metadata, bitstreams)

    original_size = image.size * image.dtype.itemsize
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size

    print(f'Input image shape: {image.shape}')
    print(f'Original uncompressed bytes: {original_size:,}')
    print(f'Compressed file bytes: {compressed_size:,}')
    print(f'Compression ratio: {compression_ratio:.2f}:1')

    return output_path


if __name__ == '__main__':
    compress_with_dct_and_arithmetic()

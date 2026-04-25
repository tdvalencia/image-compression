import codec.tools as ct
import numpy as np
from scipy.fft import dctn
import codec.encoders.run_length as rle
import codec.encoders.huffman as hf
import os

BLOCK_SIZE = 8

def build_quantization_matrix(block_size=8):
    q_matrix = np.zeros((block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            # Quadratic scaling: Low frequencies stay below 10,000, 
            # High frequencies skyrocket into the 100,000s
            q_matrix[i, j] = 1000 + ((i + j)**2) * 2500
    q_matrix[0, 0] = 500
    return q_matrix

def compress_with_dct_and_hf(input_path='images/rgb16bit/deer.ppm', output_path='deer_compressed.uofm'):
    # format image to be divisible by block size and force 3 color channels (RGB)
    image = ct.load_and_preprocess_image(input_path, block_size=BLOCK_SIZE)
    # image format: (num_blocks_y, num_blocks_x, channels, block_size, block_size)
    blocked_image = ct.block_image(image, block_size=BLOCK_SIZE)

    # takes 2D dct of each block axes=(3, 4) (2D DCT applied to each color channel)
    # norm='ortho' does 1/sqrt(N) normalization forward and assumes 1/sqrt(N) normalization backward
    # similar to the DFT definition we learned beginning of semester
    dct_blocks = np.asarray(dctn(blocked_image, axes=(3, 4), norm='ortho'))

    q_matrix = build_quantization_matrix(BLOCK_SIZE)
    quantized_dct_blocks = np.round(dct_blocks / q_matrix).astype(np.int32)

    # mask to keep the top-left NxN coefficients (lowest frequency) of each block
    N = 8
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    mask[:N, :N] = 1

    # apply the mask (NumPy broadcasting automatically applies the 2D mask to all 3 color channels)
    masked_dct_blocks = quantized_dct_blocks * mask
    flattened_blocks = masked_dct_blocks.reshape(-1, BLOCK_SIZE, BLOCK_SIZE)

    # zig-zag flattened and RLE encode each block of DCT coefficients
    rle_blocks = []
    for block in flattened_blocks:
        flat_zig_zag = rle.zigzag_flatten(block)
        rle_blocks.extend(rle.encode(flat_zig_zag))

    print(f"Total RLE tuples generated: {len(rle_blocks)}")

    # entropy encode the RLE output
    compressed_bits, symbol_counts = hf.encode_rle(rle_blocks)
    saved_metadata = {
        'symbol_counts': symbol_counts
    }

    # save the compressed bits and metadata into our universal container format
    ct.save_uofm_container(output_path, image.shape, 'huffman', saved_metadata, compressed_bits)

# Calculate compression stats
    original_size = image.size * image.dtype.itemsize
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    print(f'Original size: {original_size} bytes')
    print(f'Compressed size: {compressed_size} bytes')
    print(f'Compression ratio: {compression_ratio:.2f}')

if __name__ == '__main__':
    compress_with_dct_and_hf()
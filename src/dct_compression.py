import src.codec_tools as ct
import numpy as np
from scipy.fft import dctn
import src.encoders.run_length as rle
import src.encoders.arithmetic as ae

BLOCK_SIZE = 8

if __name__ == '__main__':
    # format image to be divisible by block size and force 3 color channels (RGB)
    image = ct.load_and_preprocess_image('images/rgb16bit/deer.ppm', block_size=BLOCK_SIZE)
    # image format: (num_blocks_y, num_blocks_x, channels, block_size, block_size)
    blocked_image = ct.block_image(image, block_size=BLOCK_SIZE)

    # takes 2D dct of each block axes=(3, 4) (2D DCT applied to each color channel)
    # norm='ortho' does 1/sqrt(N) normalization forward and assumes 1/sqrt(N) normalization backward
    # similar to the DFT definition we learned beginning of semester
    dct_blocks = np.asarray(dctn(blocked_image, axes=(3, 4), norm='ortho'))
    quantized_dct_blocks = np.round(dct_blocks).astype(int)

    # mask to keep the top-left NxN coefficients (lowest frequency) of each block
    N = 1
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    mask[:N, :N] = 1

    # apply the mask (NumPy broadcasting automatically applies the 2D mask to all 3 color channels)
    masked_dct_blocks = np.asarray(quantized_dct_blocks) * mask
    block_matrix = masked_dct_blocks.reshape(-1, BLOCK_SIZE, BLOCK_SIZE)

    # zig-zag flattened and RLE encode each block of DCT coefficients
    rle_blocks = []
    for block in block_matrix:
        flat_zig_zag = rle.zigzag_flatten(block)
        rle_blocks.extend(rle.encode(flat_zig_zag))

    print(f"Total RLE tuples generated: {len(rle_blocks)}")

    # arithmetic encode the RLE output
    compressed_bits, probabilites, total_symbols = ae.encode_rle(rle_blocks)
    saved_metadata = {
        'probabilities': probabilites,
        'total_symbols': total_symbols
    }

    # save the compressed bits and metadata into our universal container format
    ct.save_uofm_container('deer_compressed.uofm', image.shape, 'arithmetic', saved_metadata, compressed_bits)
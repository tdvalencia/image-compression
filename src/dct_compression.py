import codec_tools as ct
import numpy as np
from scipy.fftpack import dctn, idctn
from PIL import Image

BLOCK_SIZE = 8

if __name__ == '__main__':
    # format image to be divisible by block size and force 3 color channels (RGB)
    image = ct.load_and_preprocess_image('images/rgb16bit/deer.ppm', block_size=BLOCK_SIZE)
    # image format: (num_blocks_y, num_blocks_x, channels, block_size, block_size)
    blocked_image = ct.block_image(image, block_size=BLOCK_SIZE)

    # takes 2D dct of each block axes=(3, 4) (2D DCT applied to each color channel)
    # norm='ortho' does 1/sqrt(N) normalization forward and assumes 1/sqrt(N) normalization backward
    # similar to the DFT definition we learned beginning of semester
    dct_blocks = dctn(blocked_image, axes=(3, 4), norm='ortho')

    # mask to keep the top-left NxN coefficients (lowest frequency) of each block
    N = 1
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    mask[:N, :N] = 1

    # apply the mask (NumPy broadcasting automatically applies the 2D mask to all 3 color channels)
    masked_dct_blocks = dct_blocks * mask

    # Inverse DCT
    idct_blocks = idctn(masked_dct_blocks, axes=(3, 4), norm='ortho')

    # Reshape back to the standard 3D image format (H, W, C)
    H, W, C = image.shape
    reconstructed_image = idct_blocks.transpose(0, 3, 1, 4, 2).reshape(H, W, C)

    ct.plot_zoomed_comparison(image, reconstructed_image, title=f'Compressed Image (N={N})')

    img = Image.fromarray((reconstructed_image * 255).astype(np.uint8))
    img.save('images/compressed/deer_compressed.png')

    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save('images/compressed/deer_original.png')
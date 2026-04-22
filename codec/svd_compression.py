
import codec.tools as ct
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
import codec.encoders.run_length as rle
import codec.encoders.arithmetic as ae
import codec.encoders.huffman as hf

if __name__ == '__main__':
    # Load image
    img: np.ndarray = io.imread('theory/deer.ppm')

    # If the image is grayscale, convert it to RGB so our 3-channel math works
    if img.ndim == 2:
        img = color.gray2rgb(img)
    # If the image has an alpha channel (RGBA), strip it to standard RGB
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    img = util.img_as_uint(img)


    # Ensure image is (H, W, 3)
    image = img.astype(np.float64)

    k = 200  # compression level (tune this)

    compressed_channels = []
    reconstructed_channels = []
    flat_data = []

    # Process each RGB channel independently
    for c in range(3):
        channel = image[:, :, c]

        # Perform truncated SVD
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)

        # Keep only top-k singular values
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        compressed_channels.append((U_k, S_k, Vt_k))

        # Reconstruct channel from truncated SVD
        reconstructed = U_k @ np.diag(S_k) @ Vt_k
        reconstructed_channels.append(reconstructed)

        # Quantization
        U_k = np.round(U_k, 4)
        S_k = np.round(S_k, 2)
        Vt_k = np.round(Vt_k, 4)

        # Flatten and append
        flat_data.extend(U_k.flatten())
        flat_data.extend(S_k)
        flat_data.extend(Vt_k.flatten())

    # Stack reconstructed channels back
    reconstructed_image = np.stack(reconstructed_channels, axis=2)

    # Clip values to valid image range
    reconstructed_image = np.clip(reconstructed_image, 0, 65535).astype(np.uint16)

    display_reconstructed = reconstructed_image.astype(np.float32) / 65535.0
    ct.plot_zoomed_comparison(image / 65535.0, display_reconstructed, title='Reconstructed Compressed Image')

    # Save output
    rle_img = rle.encode(flat_data)

    print(f"Total RLE tuples generated: {len(rle_img)}")

    # entropy encode the RLE output
    compressed_bits, symbol_counts = hf.encode_rle(rle_img)
    saved_metadata = {
        'symbol_counts': symbol_counts
    }

    # save the compressed bits and metadata into our universal container format
    ct.save_uofm_container('deer_compressed.uofm', image.shape, 'huffman', saved_metadata, compressed_bits)
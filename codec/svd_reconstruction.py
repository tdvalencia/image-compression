import codec.tools as ct
import numpy as np
from skimage import io, color, util
import codec.encoders.run_length as rle
import codec.metrics as metrics

if __name__ == '__main__':
    # 1. Load the metadata and the RLE tuples from the custom file
    # shape is the original image tuple, e.g., (Height, Width, Channels)
    shape, decoded_rle_tuples = ct.load_uofm_container('deer_compressed.uofm')
    H, W, C = shape
    k = 400

    # 2. RLE DECODE
    # Expand [(1, 145), (63, 0)] back out into [145, 0, 0, 0...]
    # (Assuming you have an rle.decode function to reverse rle.encode)
    flat_data = rle.decode_master_rle_list(decoded_rle_tuples)

    print(f"Total frequencies decoded: {len(flat_data)}")

    # 3. RECONSTRUCT CHANNELS
    reconstructed_channels = []
    idx = 0

    for c in range(C):
        # U
        U_size = H * k
        U_k = np.array(flat_data[idx:idx + U_size]).reshape(H, k)
        idx += U_size

        # S
        S_k = np.array(flat_data[idx:idx + k])
        idx += k

        # Vt
        Vt_size = k * W
        Vt_k = np.array(flat_data[idx:idx + Vt_size]).reshape(k, W)
        idx += Vt_size

        # Reconstruct channel
        channel = U_k @ np.diag(S_k) @ Vt_k
        reconstructed_channels.append(channel)

    # --- Combine channels ---
    reconstructed_image = np.stack(reconstructed_channels, axis=2)

    # Clip values to valid image range
    reconstructed_image = np.clip(reconstructed_image, 0, 65535).astype(np.uint16)

    # 4. LOAD ORIGINAL IMAGE EARLY (We need it to check the scale!)
    # Load image
    img: np.ndarray = io.imread('theory/deer.ppm')

    # If the image is grayscale, convert it to RGB so our 3-channel math works
    if img.ndim == 2:
        img = color.gray2rgb(img)
    # If the image has an alpha channel (RGBA), strip it to standard RGB
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    original_image = util.img_as_uint(img)

    # 5. EVALUATE METRICS
    original_float_norm = original_image.astype(np.float32) / 65535.0
    reconstructed_float_norm = reconstructed_image.astype(np.float32) / 65535.0

    psnr_value = metrics.psnr(original_float_norm, reconstructed_float_norm)
    ssim_value = metrics.ssim(original_float_norm, reconstructed_float_norm)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

    # 6. VIEW RESULTS
    # Matplotlib cannot render uint16. Using normalized floats above for display.
    ct.plot_zoomed_comparison(original_float_norm, reconstructed_float_norm, title='Reconstructed Compressed Image')
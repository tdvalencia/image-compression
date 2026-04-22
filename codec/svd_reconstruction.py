import codec.tools as ct
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
import codec.encoders.run_length as rle
import codec.metrics as metrics

if __name__ == '__main__':
    # 1. Load the metadata and the RLE tuples from the custom file
    # shape is the original image tuple, e.g., (Height, Width, Channels)
    shape, decoded_rle_tuples = ct.load_uofm_container('deer_compressed.uofm')
    H, W, C = shape
    k = 200

    # 2. RLE DECODE
    # Expand [(1, 145), (63, 0)] back out into [145, 0, 0, 0...]
    # (Assuming you have an rle.decode function to reverse rle.encode)
    flat_data = rle.decode_master_rle_list(decoded_rle_tuples)

    print(f"Total frequencies decoded: {len(flat_data)}")

    # 3. INVERSE ZIG-ZAG
    #reconstructed_image = rle.unflatten(flat_data)

    #print(f"Max reconstructed pixel: {reconstructed_image.max()}")

    # 4. RECONSTRUCT CHANNELS
    reconstructed_channels = []
    idx = 0

    for c in range(C):
        # U
        U_size = H * k
        U_k = flat_data[idx:idx + U_size]
        np.array(U_k).reshape(H, k)
        idx += U_size

        # S
        S_k = flat_data[idx:idx + k]
        np.array(S_k)
        idx += k

        # Vt
        Vt_size = k * W
        Vt_k = flat_data[idx:idx + Vt_size]
        np.array(Vt_k).reshape(H, k).reshape(k, W)
        idx += Vt_size

        # Reconstruct channel
        channel = U_k @ np.diag(S_k) @ Vt_k
        reconstructed_channels.append(channel)

    # --- Combine channels ---
    reconstructed_image = np.stack(reconstructed_channels, axis=2)


    # 5. LOAD ORIGINAL IMAGE EARLY (We need it to check the scale!)
    # Load image
    img: np.ndarray = io.imread('theory/deer.ppm')

    # If the image is grayscale, convert it to RGB so our 3-channel math works
    if img.ndim == 2:
        img = color.gray2rgb(img)
    # If the image has an alpha channel (RGBA), strip it to standard RGB
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)

    original_image = util.img_as_uint(img)


    # Ensure image is (H, W, 3)
    original_image = img.astype(np.float64)
    original_image = original_image[:H, :W, :]

    # 6. EVALUATE METRICS
    psnr_value = metrics.psnr(original_image, reconstructed_image)
    ssim_value = metrics.ssim(original_image, reconstructed_image)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

    # 7. VIEW RESULTS
    # Matplotlib cannot render uint16. We must convert the arrays to 0.0 - 1.0 floats strictly for the plotter!
    display_original = original_image.astype(np.float32) / 65535.0
    display_reconstructed = reconstructed_image.astype(np.float32) / 65535.0
    ct.plot_zoomed_comparison(display_original, display_reconstructed, title='Reconstructed Compressed Image')
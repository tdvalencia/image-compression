'''
    Forces JPEG to match the compressed file size of our custom codec, then compares PSNR and SSIM between the two reconstructions.
    This is a more direct "apples-to-apples" comparison than just picking a JPEG.
'''

import os
import glob
import io
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric

# custom imports
import codec.tools as ct
import codec.pipelines.hybrid_hf_compression as custom_codec

IMAGE_DIR = 'images/rgb16bit/'
TEMP_OUT = 'temp_batch_custom.uofm'

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(original, compressed):
    return ssim_metric(original, compressed, data_range=1.0, channel_axis=2, win_size=5)

def run_directory_benchmark():
    ppm_files = glob.glob(os.path.join(IMAGE_DIR, "*.ppm"))
    
    if not ppm_files:
        print(f"No .ppm files found in {IMAGE_DIR}. Please check the path.")
        return

    print(f"--- COMPREHENSIVE RATE-DISTORTION BENCHMARK ---")
    print(f"Target Directory: {IMAGE_DIR}")
    print(f"Total Images: {len(ppm_files)}\n")
    
    # Print the wider table header
    header = f"{'Image Name':<18} | {'Size KB (Cust / JPEG)':<22} | {'PSNR (Cust / JPEG)':<18} | {'SSIM (Cust / JPEG)':<18} | {'JPEG(Q)'}"
    print(header)
    print("-" * len(header))

    # Track overall winners
    tally = {'custom_ssim': 0, 'jpeg_ssim': 0, 'custom_psnr': 0, 'jpeg_psnr': 0}

    for img_path in ppm_files:
        base_name = os.path.basename(img_path)
        
        # 1. LOAD ORIGINAL
        original_16bit = ct.load_and_preprocess_image(img_path)
        original_float = original_16bit.astype(np.float32) / 65535.0

        # 2. RUN CUSTOM PIPELINE
        custom_codec.compress_image(img_path, TEMP_OUT)
        custom_bytes = os.path.getsize(TEMP_OUT)
        
        reconstructed_16bit = custom_codec.reconstruct_image(TEMP_OUT)
        custom_float = reconstructed_16bit.astype(np.float32) / 65535.0
        
        custom_ssim = calculate_ssim(original_float, custom_float)
        custom_psnr = calculate_psnr(original_float, custom_float)

        # 3. RUN JPEG MATCHING LOOP
        original_8bit = (original_float * 255).astype(np.uint8)
        pil_image = Image.fromarray(original_8bit)
        
        best_jpeg_quality = 1
        closest_size_diff = float('inf')
        best_jpeg_buffer = None

        for q in range(1, 101):
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=q)
            size = buffer.tell()
            
            diff = abs(size - custom_bytes)
            if diff < closest_size_diff:
                closest_size_diff = diff
                best_jpeg_quality = q
                best_jpeg_buffer = buffer

        # Decode winning JPEG
        best_jpeg_buffer.seek(0)
        jpeg_bytes = best_jpeg_buffer.getbuffer().nbytes
        jpeg_reconstructed = np.array(Image.open(best_jpeg_buffer))
        jpeg_float = jpeg_reconstructed.astype(np.float32) / 255.0

        # Ensure dimensions match for metric calculations
        H = min(original_float.shape[0], jpeg_float.shape[0])
        W = min(original_float.shape[1], jpeg_float.shape[1])
        original_cropped = original_float[:H, :W, :]
        jpeg_cropped = jpeg_float[:H, :W, :]

        jpeg_ssim = calculate_ssim(original_cropped, jpeg_cropped)
        jpeg_psnr = calculate_psnr(original_cropped, jpeg_cropped)

        # 4. TRACK WINNERS
        if custom_ssim > jpeg_ssim: tally['custom_ssim'] += 1
        else: tally['jpeg_ssim'] += 1

        if custom_psnr > jpeg_psnr: tally['custom_psnr'] += 1
        else: tally['jpeg_psnr'] += 1

        # 5. PRINT ROW
        size_str = f"{custom_bytes/1024:.1f} / {jpeg_bytes/1024:.1f}"
        psnr_str = f"{custom_psnr:.2f} / {jpeg_psnr:.2f}"
        ssim_str = f"{custom_ssim:.4f} / {jpeg_ssim:.4f}"
        
        print(f"{base_name[:18]:<18} | {size_str:<22} | {psnr_str:<18} | {ssim_str:<18} | Q={best_jpeg_quality}")

    print("-" * len(header))
    print("FINAL TALLY:")
    print(f"  SSIM Wins -> Custom: {tally['custom_ssim']} | JPEG: {tally['jpeg_ssim']}")
    print(f"  PSNR Wins -> Custom: {tally['custom_psnr']} | JPEG: {tally['jpeg_psnr']}")

    # Clean up
    if os.path.exists(TEMP_OUT):
        os.remove(TEMP_OUT)

if __name__ == '__main__':
    run_directory_benchmark()
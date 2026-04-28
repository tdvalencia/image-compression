'''
    Compression pipeline that utilizes PCA, DPCM, DCT, quantization, and Huffman coding.
    This pipeline is to be the best performing one and hopefully beats the JPEG encoding.
'''

import numpy as np
from scipy.fft import dctn, idctn
import codec.tools as ct
import codec.encoders.pca as pca
import codec.encoders.dpcm as dpcm
import codec.encoders.run_length as rle
import codec.encoders.huffman as hf
import codec.metrics as metrics
import os

def compress_image(image_path, output_path):
    # 1. Load image (H, W, 3)
    img = ct.load_and_preprocess_image(image_path)
    
    # 2. COLOR DECORRELATION (PCA)
    # Pass the ENTIRE image so it can rotate the 3 color channels against each other
    pca_image, Vt, means = pca.apply_pca_prefilter(img)

    # 3. SPATIAL BLOCKING
    # Chop the Eigen-channels into 8x8 blocks
    blocked_image = ct.block_image(pca_image, block_size=8)

    # 4. SPATIAL DECORRELATION (DCT)
    # Convert spatial blocks into frequency domain
    dct_blocks = np.asarray(dctn(blocked_image, axes=(3, 4), norm='ortho'))

    # 5. PCA-AWARE QUANTIZATION
    # We build a base matrix, then scale it based on the channel's importance
    base_q_matrix = ct.build_quantization_matrix(8)
    
    quantized_blocks = np.zeros_like(dct_blocks)
    
    # Process each PCA channel separately
    for c in range(3):
        if c == 0:
            # PC1 (Luminance): Use the base matrix to preserve detail
            channel_q_matrix = base_q_matrix
        else:
            # PC2 & PC3 (Chrominance): Crush it. Multiply the penalty by 4x to 8x
            channel_q_matrix = base_q_matrix * 6.0 
            
        # Quantize this specific channel
        quantized_blocks[:, :, c, :, :] = np.round(
            dct_blocks[:, :, c, :, :] / channel_q_matrix
        ).astype(np.int32)

    # 6. SERIALIZATION & STREAM SEPARATION
    flattened_blocks = quantized_blocks.reshape(-1, 8, 8)
    raw_dc_stream = []
    ac_rle_tuples = []

    for block in flattened_blocks:
        flat_zigzag = rle.zigzag_flatten(block)
        
        # Pull out the DC coefficient
        raw_dc_stream.append(flat_zigzag[0])
        
        # RLE encode the remaining AC coefficients
        ac_only_array = flat_zigzag[1:]
        ac_rle_tuples.extend(rle.encode(ac_only_array)) # get rle tuples for AC stream

    # 7. PREDICTIVE ENCODING (DPCM)
    # Now we apply DPCM exclusively to the sequence of DC coefficients
    dc_deltas = dpcm.apply_dpcm(raw_dc_stream)

    # 8. ENTROPY ENCODING (Huffman)
    # Generate the two optimized bitstreams
    dc_bytes, dc_counts = hf.encode(dc_deltas)
    ac_bytes, ac_counts = hf.encode(ac_rle_tuples)
    
    # 9. SAVE TO DISK
    metadata = {
        'pca_Vt': Vt,
        'pca_means': means,
        'dc_symbol_counts': dc_counts,
        'ac_symbol_counts': ac_counts,
        'image_shape': img.shape
    }

    # dahuffman (backend) gives us the raw bytes, no need
    # to do any additional packing ourselves. Just save the bytes and the metadata.
    bitstreams = {
        'dc_bits': dc_bytes,
        'ac_bits': ac_bytes
    }

    ct.save_uofm_container(output_path, img.shape, metadata, bitstreams)

def reconstruct_image(compressed_path) -> np.ndarray:
    # 1. Load the metadata and the bitstreams from the custom file
    shape, metadata, bitstreams = ct.load_uofm_container(compressed_path)
    H, W, C = shape
    Vt = metadata['pca_Vt']
    means = metadata['pca_means']
    dc_counts = metadata['dc_symbol_counts']
    ac_counts = metadata['ac_symbol_counts']
    dc_bytes = bitstreams['dc_bits']
    ac_bytes = bitstreams['ac_bits']

    # 2. ENTROPY DECODING (Huffman)
    dc_deltas = hf.decode(dc_bytes, dc_counts)
    ac_rle_tuples = hf.decode(ac_bytes, ac_counts)

    # 3. INVERSE PREDICTIVE ENCODING (DPCM)
    raw_dc_stream = dpcm.invert_dpcm(dc_deltas)

    # 4. INVERSE RLE
    # We need to reconstruct the original sequence of quantized DCT blocks.
    # Start by rebuilding the DC + AC zig-zag arrays for each block.
    flat_ac_frequencies = rle.decode_master_rle_list(ac_rle_tuples)
    num_blocks = len(raw_dc_stream)

    # 5. STREAM RECOMBINATION & INVERSE ZIG-ZAG
    chunked_ac = np.array(flat_ac_frequencies).reshape(num_blocks, 63) # 63 AC coefficients per block
    reconstructed_blocks = []

    for i in range(num_blocks):
        full_64_zigzag = np.concatenate(([raw_dc_stream[i]], chunked_ac[i]))
        block_8x8 = rle.zigzag_unflatten(full_64_zigzag)
        reconstructed_blocks.append(block_8x8)

    # Convert the flat list into a 3D tensor (N, 8, 8)
    flat_quantized_blocks = np.array(reconstructed_blocks)

    num_blocks_y = H // 8
    num_blocks_x = W // 8
    quantized_blocks = flat_quantized_blocks.reshape(num_blocks_y, num_blocks_x, C, 8, 8)

    # 6. INVERSE QUANTIZATION (PCA-Aware)
    # Rebuild the exact same base matrix used in the compressor
    base_q_matrix = ct.build_quantization_matrix(8)
    
    # Create an empty tensor to hold the restored frequencies
    dequantized_blocks = np.zeros_like(quantized_blocks, dtype=np.float32)

    # Process each PCA channel with its specific multiplier
    for c in range(3):
        if c == 0:
            # PC1 (Luminance): Multiplied by the base matrix
            channel_q_matrix = base_q_matrix
        else:
            # PC2 & PC3 (Chrominance): Multiplied by the heavily crushed matrix
            # IMPORTANT: This multiplier must identically match your compression script!
            channel_q_matrix = base_q_matrix * 6.0 
            
        # Multiply the quantized integers by the matrix to restore the magnitudes
        dequantized_blocks[:, :, c, :, :] = quantized_blocks[:, :, c, :, :] * channel_q_matrix

    # 7. INVERSE SPATIAL DECORRELATION (IDCT)
    pca_pixel_blocks = np.asarray(idctn(dequantized_blocks, axes=(-2, -1), norm='ortho'))
    
    # 8. REBUILD IMAGE
    num_blocks_y = H // 8
    num_blocks_x = W // 8
    blocks_reshaped = pca_pixel_blocks.reshape(num_blocks_y, num_blocks_x, C, 8, 8)
    pca_image = ct.unblock_image(blocks_reshaped, shape)

    # 9. INVERSE COLOR DECORRELATION (PCA)
    final_rgb_image = pca.invert_pca(pca_image, Vt, means)
    
    return final_rgb_image

if __name__ == '__main__':
    original_img = ct.load_and_preprocess_image('theory/deer.ppm')
    compressed_file = 'deer_compressed.uofm'

    print('Compressing image...')
    compress_image('theory/deer.ppm', compressed_file)

    print('Reconstructing image from compressed file...')
    reconstructed_img = reconstruct_image(compressed_file)

    # Evaluate metrics
    print('Computing metrics...')
    float_original = original_img.astype(np.float32) / 65535.0
    float_reconstructed = reconstructed_img.astype(np.float32) / 65535.0

    print(f"float_original dtype: {float_original.dtype}, range: [{float_original.min():.4f}, {float_original.max():.4f}]")
    print(f"float_reconstructed dtype: {float_reconstructed.dtype}, range: [{float_reconstructed.min():.4f}, {float_reconstructed.max():.4f}]")
    print(f"MSE: {np.mean((float_original - float_reconstructed)**2):.6f}")

    # Manual PSNR calculation
    mse = np.mean((float_original - float_reconstructed)**2)
    manual_psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"Manual PSNR calculation: {manual_psnr:.2f} dB")

    psnr_value = metrics.psnr(float_original, float_reconstructed)
    ssim_value = metrics.ssim(float_original, float_reconstructed)

    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'SSIM: {ssim_value:.4f}')

    original_size = original_img.size * original_img.dtype.itemsize
    compressed_size = os.path.getsize(compressed_file)
    compression_ratio = original_size / compressed_size

    print(f'Input image shape: {original_img.shape}')
    print(f'Original uncompressed bytes: {original_size:,}')
    print(f'Compressed file bytes: {compressed_size:,}')
    print(f'Compression ratio: {compression_ratio:.2f}:1')

    ct.plot_zoomed_comparison(float_original, float_reconstructed, title="Hybrid HF Compression")

    import matplotlib.pyplot as plt

    # Compute the absolute difference
    diff = np.abs(float_original - float_reconstructed)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(float_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Reconstructed
    axes[1].imshow(float_reconstructed)
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    # Difference (scaled for visibility)
    diff_display = diff / diff.max()  # Normalize to 0-1 for visibility
    axes[2].imshow(diff_display, cmap='hot')
    axes[2].set_title(f'Absolute Difference\n(Max: {diff.max():.4f}, Mean: {diff.mean():.4f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('difference_map.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Optional: Print statistics
    print(f"Difference Statistics:")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std: {diff.std():.6f}")
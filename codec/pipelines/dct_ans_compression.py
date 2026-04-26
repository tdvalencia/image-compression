import numpy as np
import codec.tools as ct
import codec.encoders.ans as ans
import codec.encoders.run_length as rle
from scipy.fft import dctn
import codec.metrics as metrics
import os

BLOCK_SIZE = 8
INPUT_IMAGE = 'images/rgb16bit/deer.ppm'
OUTPUT_FILE = 'deer_ans_compressed.uofm'

def build_quantization_matrix(block_size=8):
    q_matrix = np.zeros((block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            # Quadratic scaling: Low frequencies stay below 10,000, 
            # High frequencies skyrocket into the 100,000s
            q_matrix[i, j] = 1000 + ((i + j)**2) * 2500
    q_matrix[0, 0] = 500
    return q_matrix

def compress_with_dct_and_ans(input_path=INPUT_IMAGE, output_path=OUTPUT_FILE):
    image = ct.load_and_preprocess_image(input_path, block_size=BLOCK_SIZE)
    blocked_image = ct.block_image(image, block_size=BLOCK_SIZE)

    dct_blocks = np.asarray(dctn(blocked_image, axes=(3, 4), norm='ortho'))

    q_matrix = build_quantization_matrix(BLOCK_SIZE)
    quantized_blocks = np.round(dct_blocks / q_matrix).astype(np.int32)

    # mask blocks to keep only the top-left NxN coefficients (lowest frequency)
    N = 8
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    mask[:N, :N] = 1
    masked_blocks = quantized_blocks * mask

    flattened_blocks = masked_blocks.reshape(-1, BLOCK_SIZE, BLOCK_SIZE)

    rle_tuples = []
    for block in flattened_blocks:
        flat_zigzag = rle.zigzag_flatten(block)
        rle_tuples.extend(rle.encode(flat_zigzag))

    print(f'Total RLE tuples generated: {len(rle_tuples)}')

    encoded_signal = ans.encode_rle(rle_tuples)
    # print(f"Words type: {type(encoded_signal.words)}")
    # print(f"Words dtype: {encoded_signal.words.dtype}")
    # print(f"Words shape: {encoded_signal.words.shape}")
    # print(f"Words length: {len(encoded_signal.words)}")

    encoded_bytes = encoded_signal.words.tobytes()
    # print(f"Encoded bytes length: {len(encoded_bytes)}")
    # print(f"Expected (words_length * 8): {len(encoded_signal.words) * 8}")
    # print(f"Actual % 8: {len(encoded_bytes) % 8}")

    ans_metadata = {
        'state': int(encoded_signal.state),
        'counts': encoded_signal.symbol_counts,
        'values': encoded_signal.symbol_values,
        'length': encoded_signal.signal_length,
        'words_length': len(encoded_signal.words),  # number of uint32 code words in the stream
        'bytes_length': len(encoded_bytes)  # Store the exact byte length of the encoded data
    }

    # print(f"Words length: {len(encoded_signal.words)}, bytes length: {len(encoded_bytes)}")

    bitstreams = {'ans_words': encoded_bytes}
    ct.save_uofm_container(output_path, image.shape, ans_metadata, bitstreams)

    original_size = image.size * image.dtype.itemsize
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    print(f'Original size: {original_size} bytes')
    print(f'Compressed size: {compressed_size} bytes')
    print(f'Compression ratio: {compression_ratio:.2f}')

if __name__ == '__main__':
    compress_with_dct_and_ans()
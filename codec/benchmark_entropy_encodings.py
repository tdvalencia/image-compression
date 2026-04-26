import os
import glob
import pickle
import numpy as np
import codec.tools as ct
from codec.pipelines.dct_ans_compression import compress_with_dct_and_ans
from codec.pipelines.dct_hf_compression import compress_with_dct_and_hf

# Configuration
IMAGE_DIR = 'images/rgb16bit/'
ANS_OUT_DIR = 'compressed_outputs/ans/'
HF_OUT_DIR = 'compressed_outputs/huffman/'

os.makedirs(ANS_OUT_DIR, exist_ok=True)
os.makedirs(HF_OUT_DIR, exist_ok=True)

def get_metadata_size(uofm_path):
    """
    Opens the container and measures the byte-size of the metadata dictionary.
    """
    with open(uofm_path, 'rb') as f:
        container = pickle.load(f)
    # Measure the size of the metadata portion specifically
    return len(pickle.dumps(container['encoder_metadata']))

def run_benchmark():
    ppm_files = glob.glob(os.path.join(IMAGE_DIR, "*.ppm"))
    
    # Updated header for detailed audit
    header = f"{'Image Name':<18} | {'Type':<7} | {'Total(KB)':<10} | {'Meta(KB)':<10} | {'Stream(KB)':<10} | {'Ratio':<7}"
    print(header)
    print("-" * len(header))

    for img_path in ppm_files:
        base_name = os.path.basename(img_path)
        ans_path = os.path.join(ANS_OUT_DIR, base_name.replace('.ppm', '_ans.uofm'))
        hf_path = os.path.join(HF_OUT_DIR, base_name.replace('.ppm', '_hf.uofm'))

        # Run Compressions
        compress_with_dct_and_ans(img_path, ans_path)
        compress_with_dct_and_hf(img_path, hf_path)

        # Get Raw Stats
        temp_img = ct.load_and_preprocess_image(img_path, block_size=8)
        orig_size = temp_img.size * 2 # 16-bit
        
        for method, path in [("ANS", ans_path), ("Huff", hf_path)]:
            total_size = os.path.getsize(path)
            meta_size = get_metadata_size(path)
            stream_size = total_size - meta_size # Approximation of the bitstream
            ratio = orig_size / total_size

            print(f"{base_name[:18]:<18} | {method:<7} | {total_size/1024:>9.1f} | {meta_size/1024:>9.1f} | {stream_size/1024:>10.1f} | {ratio:>6.1f}:1")
        
        print("-" * len(header))

if __name__ == '__main__':
    run_benchmark()
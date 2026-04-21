'''
    Arithmetic coding implementation for compressing RLE data from DCT blocks.
    This module provides functions to encode RLE data into a compressed bitstream
    and to decode it back to the original RLE format.

    Does not handle large images well, we are running into floating point precision
    issues with the probabilities.
'''

from collections import Counter
from arithmetic_compressor import AECompressor
from arithmetic_compressor.util import Range
from arithmetic_compressor.models import StaticModel
import numpy as np

class ExactStaticModel:
    def __init__(self, symbol_counts):
        self.symbol_counts = dict(symbol_counts)
        self.total = sum(symbol_counts.values())
        # Use np.float64 or np.float128 for higher precision
        self.__prob = {sym: np.float64(cnt) / np.float64(self.total) for sym, cnt in self.symbol_counts.items()}
        self.cdf_object = {}
        prev = 0
        for sym, cnt in self.symbol_counts.items():
            self.cdf_object[sym] = Range(prev, prev + cnt)
            prev += cnt

    def cdf(self):
        return self.cdf_object

    def probability(self):
        return self.__prob

SCALE_FACTOR = 65536

def encode_rle(rle_data):
    # 1. Format data securely into value_count
    string_symbols = [f'{val}_{count}' for val, count in rle_data]
    
    # 2. Build exact symbol counts from the RLE stream
    raw_counts = Counter(string_symbols)
    total = sum(raw_counts.values())

    # 3. Scale counts to a fixed total (65536) for better precision
    SCALE_TOTAL = 65536
    symbol_counts = {}
    for sym, cnt in raw_counts.items():
        symbol_counts[sym] = max(1, round(cnt / total * SCALE_TOTAL))

    # 4. Pass scaled counts to ExactStaticModel, compress the sequence
    model = ExactStaticModel(symbol_counts)
    coder = AECompressor(model, adapt=False)
    compressed_bits = coder.compress(string_symbols)
    
    return compressed_bits, symbol_counts, len(string_symbols)


def decode_rle(compressed_bits, symbol_counts, total_symbols):
    '''
    Inverse of encode_rle.
    Returns RLE data as a list of (value, count) tuples.
    '''
    # 1. Load the same static model used for encoding
    model = ExactStaticModel(symbol_counts)
    coder = AECompressor(model, adapt=False)
    
    # 2. Decompress bits back to 'value_count' strings
    string_symbols = coder.decompress(compressed_bits, total_symbols)

    # 3. Parse strings straight back into tuples
    rle_data = []
    for sym in string_symbols:
        val_str, count_str = sym.split('_')
        rle_data.append((int(val_str), int(count_str)))
            
    return rle_data


if __name__ == '__main__':
    # 1. Create a fake block of RLE data
    # using (value, count) tuple format
    mock_rle_data = [
        (150, 1),    # The DC coefficient
        (-12, 1),    # A small AC detail
        (0, 14),     # A run of 14 zeros
        (4, 1),      # Another AC detail
        (0, 47)      # The trailing zeros completing the 64-item block
    ]

    print("--- 1. ORIGINAL DATA ---")
    print(mock_rle_data)
    print(f"Original Length: {len(mock_rle_data)} tuples\n")

    # 2. Run the Encoder
    print("Compressing...")
    compressed_bits, probabilities, total_symbols = encode_rle(mock_rle_data)

    print("\n--- 2. COMPRESSION STATS ---")
    # Truncating the bitstream output for readability in the console
    print(f"Bitstream Output: {str(compressed_bits)[:50]}... (truncated)")
    print(f"Physical Size: {len(compressed_bits)} bits")
    print(f"Total Symbols: {total_symbols}")
    print(f"Alphabet Size Saved: {len(probabilities)} unique symbols\n")

    # 3. Run the Decoder
    print("Decompressing...")
    reconstructed_data = decode_rle(compressed_bits, probabilities, total_symbols)

    print("\n--- 3. RECONSTRUCTED DATA ---")
    print(reconstructed_data)

    # 4. The Ultimate Proof
    print("\n--- THE VERDICT ---")
    if mock_rle_data == reconstructed_data:
        print("✅ SUCCESS! The data survived the (value, count) round trip perfectly.")
    else:
        print("❌ FAILED! The data was corrupted.")
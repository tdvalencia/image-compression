'''
    Huffman Encoder and Decoder for RLE Data
'''

from dahuffman import HuffmanCodec
from collections import Counter

def encode(data):
    '''
    Takes raw RLE data (a list of tuples) and compresses it directly
    rle_data is a list of (value, count) pairs, so we need to treat each pair as a unique symbol.
    '''
    # collect a dictionary of symbol frequencies to build Huffman tree
    symbol_counts = Counter(data)
    
    # "train" the codec and compress the data
    codec = HuffmanCodec.from_frequencies(symbol_counts)    
    compressed_bytes = codec.encode(data)
    
    return compressed_bytes, symbol_counts

def decode(compressed_bytes, symbol_counts):
    '''
    Decodes the bytes directly back into a list of tuples.
    '''
    codec = HuffmanCodec.from_frequencies(symbol_counts)
    data = codec.decode(compressed_bytes)
    
    return data

if __name__ == '__main__':
    # 1. Create a fake block of RLE data (List of tuples)
    # This simulates exactly what your RLE encoder spits out: (value, count)
    mock_rle_data = [
        (150, 1),   # The massive DC coefficient
        (-12, 1),   # A small AC detail
        (0, 14),    # A run of 14 zeros
        (4, 1),     # Another AC detail
        (0, 47)     # The trailing zeros completing the 64-item block
    ]

    mock_data = [1, 2, 3, 3, 3, 4, 4, 4, 4, 5]

    input_data = mock_rle_data  # You can switch this to fake_data to test with a different dataset

    print("--- 1. ORIGINAL DATA ---")
    print(input_data)
    print(f"Original Length: {len(input_data)} items\n")

    # 2. Run the Encoder
    compressed_bytes, saved_counts = encode(input_data)

    print("--- 2. COMPRESSION STATS ---")
    print(f"Raw Bytes Output: {compressed_bytes}")
    print(f"Physical Size: {len(compressed_bytes)} bytes")
    print(f"Huffman Dictionary Saved: {saved_counts}\n")

    # 3. Run the Decoder
    reconstructed_data = decode(compressed_bytes, saved_counts)

    print("--- 3. RECONSTRUCTED DATA ---")
    print(reconstructed_data)

    # 4. The Ultimate Proof
    print("\n--- THE VERDICT ---")
    if input_data == reconstructed_data:
        print("✅ SUCCESS! The data survived the round trip perfectly.")
    else:
        print("❌ FAILED! The data was corrupted.")
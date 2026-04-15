from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import BaseFrequencyTable

def encode_rle(rle_data):
    '''
    Takes RLE data as a list of (value, count) tuples.
    Uses an Adaptive Model with Fat Initialization.
    '''
    # 1. Format data securely into value_count
    string_symbols = [f'{val}_{count}' for val, count in rle_data]
            
    # 2. Get the unique alphabet
    unique_symbols = list(set(string_symbols))

    # 3. Fat Initialization (1000 weight) to prevent halving bug
    initial_weights = {sym: 1000 for sym in unique_symbols}
    
    # 4. Compress
    model = BaseFrequencyTable(initial_weights)
    coder = AECompressor(model)
    compressed_bits = coder.compress(string_symbols)
    
    return compressed_bits, initial_weights, len(string_symbols)


def decode_rle(compressed_bits, initial_weights, total_symbols):
    '''
    Inverse of encode_rle.
    Returns RLE data as a list of (value, count) tuples.
    '''
    # 1. Load the adaptive model
    model = BaseFrequencyTable(initial_weights)
    coder = AECompressor(model)
    
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
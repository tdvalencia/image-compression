import numpy as np
from simple_ans import ans_encode, ans_decode

def encode_rle(rle_data):
    '''
    Takes raw RLE data (a list of tuples) and compresses it directly using Asymmetric Numeral Systems (ANS)
    '''
    # Flatten the list of tuples into a single list of symbols
    symbols = [item for sublist in rle_data for item in sublist]
    
    # Convert to numpy array with appropriate dtype (int32 to handle negative DCT values)
    symbols_array = np.array(symbols, dtype=np.int32)
    
    # Encode the symbols using ANS
    compressed_bits = ans_encode(symbols_array)
    
    return compressed_bits

def decode_rle(compressed_bits):
    '''
    Takes compressed bits from ANS and decodes it back into the original RLE data (a list of tuples)
    '''
    # Decode the compressed bits back into a list of symbols
    decoded_symbols = ans_decode(compressed_bits)
    
    # Reconstruct the original RLE tuples from the flat list of symbols
    rle_tuples = []
    for i in range(0, len(decoded_symbols), 2):
        rle_tuples.append((decoded_symbols[i], decoded_symbols[i + 1]))
    
    return rle_tuples

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
    compressed = encode_rle(mock_rle_data)
    
    # Parse the EncodedSignal for inspection (optional, to understand its structure)
    print(f"EncodedSignal state: {compressed.state}")
    print(f"EncodedSignal symbol_counts: {compressed.symbol_counts}")
    print(f"EncodedSignal symbol_values: {compressed.symbol_values}")
    print(f"EncodedSignal signal_length: {compressed.signal_length}")
    print(f"EncodedSignal words: {compressed.words[:10]}... (first 10 words)")  # Truncate for readability

    # 3. Run the Decoder
    print("\nDecompressing...")
    reconstructed_data = decode_rle(compressed)

    print("\n--- 3. RECONSTRUCTED DATA ---")
    print(reconstructed_data)

    # 4. The Ultimate Proof
    print("\n--- THE VERDICT ---")
    if mock_rle_data == reconstructed_data:
        print("✅ SUCCESS! The data survived the ANS round trip perfectly.")
    else:
        print("❌ FAILED! The data was corrupted.")
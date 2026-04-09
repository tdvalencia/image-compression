from itertools import groupby
from collections import Counter
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel

def run_length_encode(data):
    """
    A highly compact, generalized RLE function.
    Works on strings, lists, tuples, or any iterable.
    """
    # groupby groups consecutive identical elements. 
    # We turn the group into a list to get its length.
    return [(len(list(group)), key) for key, group in groupby(data)]

def run_length_decode(rle_data):
    """
    Inverse of run_length_encode.
    Expects an iterable of (count, value) pairs and expands them.
    """
    out = []
    for count, val in rle_data:
        out.extend([val] * int(count))
    return out

def arithmetic_encode_rle(rle_data):
    # 1. Format the data
    # The arithmetic compressor needs discrete, hashable symbols.
    # We will convert our RLE tuples into simple strings (e.g., (1, 150) -> "1_150")
    string_symbols = [f"{count}_{val}" for count, val in rle_data]
    
    # 2. Calculate the exact probability of every symbol
    # The compressor needs to know exactly how often each symbol appears to build the math
    symbol_counts = Counter(string_symbols)
    total_symbols = len(string_symbols)
    probabilities = {symbol: count / total_symbols for symbol, count in symbol_counts.items()}
    
    # 3. Build the static probability model and the encoder
    model = StaticModel(probabilities)
    coder = AECompressor(model)
    
    # 4. Compress the data!
    # This returns a list of binary integers (0s and 1s) representing the final fractional number
    compressed_bits = coder.compress(string_symbols)
    
    return compressed_bits, probabilities, total_symbols

def arithmetic_decode_rle(compressed_bits, probabilities, total_symbols):
    """
    Inverse of arithmetic_encode_rle.
    Returns RLE data as a list of (count:int, val:int) pairs.
    """
    model = StaticModel(probabilities)
    coder = AECompressor(model)
    string_symbols = coder.decompress(compressed_bits, total_symbols)

    rle_data = []
    for sym in string_symbols:
        count_str, val_str = sym.split("_", 1)
        rle_data.append((int(count_str), int(val_str)))
    return rle_data
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel
from collections import Counter

def encode_rle(rle_data):
    '''
    Takes RLE data (a list of (val:int, count:int) pairs) and compresses it using arithmetic coding.
    Returns a tuple of (compressed_bits:list of int, probabilities:dict, total_symbols:int
    '''
    # 1. Format the data
    # The arithmetic compressor needs discrete, hashable symbols.
    # We will convert our RLE tuples into simple strings (e.g., (1, 150) -> '1_150')
    string_symbols = [f'{val}_{count}' for val, count in rle_data]
    
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

def decode_rle(compressed_bits, probabilities, total_symbols):
    '''
    Inverse of arithmetic_encode_rle.
    Returns RLE data as a list of (val:int, count:int) pairs.
    '''
    model = StaticModel(probabilities)
    coder = AECompressor(model)
    string_symbols = coder.decompress(compressed_bits, total_symbols)

    rle_data = []
    for sym in string_symbols:
        count_str, val_str = sym.split('_', 1)
        
        # 1. Safely handle the EOB string trap
        if count_str == 'EOB':
            count = 'EOB'
        else:
            count = int(count_str)
            
        # 2. The Bulletproof Value extraction
        try:
            val = int(val_str)
        except ValueError:
            # If it's '1.0' or '0.943', turn it into a float, then squash it to an int
            val = int(float(val_str)) 

        rle_data.append((val, count))
    return rle_data
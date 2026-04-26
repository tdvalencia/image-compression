'''
    Differential Pulse Code Modulation (DPCM) encoder.
'''

import numpy as np

def apply_dpcm(dc_coefficients):
    '''
    Takes a 1D array of DC coefficients.
    Returns the differential (delta) array.
    '''
    # delta[n] = current - previous
    # We pad with a 0 so the first DC is preserved (DC[0] - 0)
    deltas = np.diff(dc_coefficients, prepend=0)
    return deltas.astype(np.int32)

def invert_dpcm(deltas):
    '''
    Reconstructs DC coefficients from deltas.
    '''
    # The cumulative sum reverses the differentiation
    return np.cumsum(deltas).astype(np.int32)
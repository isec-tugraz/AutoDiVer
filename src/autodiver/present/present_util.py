import numpy as np

sbox = np.array(int(x, 16) for x in "c56b90ad3ef84712")
p_layer_order = ( 0, 16, 32, 48,  1, 17, 33, 49,  2, 18, 34, 50,  3, 19, 35, 51,
                  4, 20, 36, 52,  5, 21, 37, 53,  6, 22, 38, 54,  7, 23, 39, 55,
                  8, 24, 40, 56,  9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63)


# get permutation based on source indices (instead of target indices)
PERM = np.zeros(64, dtype=int)
PERM[np.array(p_layer_order)] = np.arange(64)

INV_PERM = np.array(p_layer_order, dtype=int)

def unpack_bits(arr: np.ndarray):
    arr = np.array(arr, dtype=np.uint8)
    bits = np.unpackbits(arr, axis=-1, bitorder='little')
    bit_offsets = [0, 1, 2, 3]
    selector = np.stack([np.arange(16) * 8 + o for o in bit_offsets]).T.flatten()
    bits = bits[..., selector]
    return bits


def pack_bits(bits: np.ndarray):
    bit_offsets = [0, 1, 2, 3]
    selector = np.stack([np.arange(16) * 8 + o for o in bit_offsets]).T.flatten()
    padded_bits = np.zeros((len(bits), 128), dtype=np.uint8)
    padded_bits[..., selector] = bits
    return np.packbits(padded_bits, axis=1, bitorder='little')


def inverse_bit_perm(arr):
    bits = unpack_bits(arr)
    ip = bits[..., INV_PERM]
    res = pack_bits(ip)
    return res


def bit_perm(arr):
    bits = unpack_bits(arr)
    pemuted = bits[..., PERM]
    res = pack_bits(pemuted)
    return res

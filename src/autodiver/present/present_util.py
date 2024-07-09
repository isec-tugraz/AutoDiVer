import numpy as np

from .present_cipher import p_layer_order

# get permutation based on source indices (instead of target indices)
PERM = np.zeros(64, dtype=int)
PERM[p_layer_order] = np.arange(64)

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

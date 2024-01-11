from typing import Literal
import numpy as np
P64 = np.array((0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3,
                4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,
                8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
                12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15))
IP64 = np.full_like(P64, -1)
IP64[P64] = np.arange(len(P64), dtype=P64.dtype)
IP64, P64 = P64, IP64
DDT = np.array([[16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  2,  2,  0,  2,  2,  2,  2,  2,  0,  0,  2],
                [ 0,  0,  0,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  2,  2,  0],
                [ 0,  0,  0,  0,  0,  2,  2,  0,  2,  0,  0,  2,  2,  2,  2,  2],
                [ 0,  0,  0,  2,  0,  4,  0,  6,  0,  2,  0,  0,  0,  2,  0,  0],
                [ 0,  0,  2,  0,  0,  2,  0,  0,  2,  0,  0,  0,  2,  2,  2,  4],
                [ 0,  0,  4,  6,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  2,  0],
                [ 0,  0,  2,  0,  0,  2,  0,  0,  2,  2,  2,  4,  2,  0,  0,  0],
                [ 0,  0,  0,  4,  0,  0,  0,  4,  0,  0,  0,  4,  0,  0,  0,  4],
                [ 0,  2,  0,  2,  0,  0,  2,  2,  2,  0,  2,  0,  2,  2,  0,  0],
                [ 0,  4,  0,  0,  0,  0,  4,  0,  0,  2,  2,  0,  0,  2,  2,  0],
                [ 0,  2,  0,  2,  0,  0,  2,  2,  2,  2,  0,  0,  2,  0,  2,  0],
                [ 0,  0,  4,  0,  4,  0,  0,  0,  2,  0,  2,  0,  2,  0,  2,  0],
                [ 0,  2,  2,  0,  4,  0,  0,  0,  0,  0,  2,  2,  0,  2,  0,  2],
                [ 0,  4,  0,  0,  4,  0,  0,  0,  2,  2,  0,  0,  2,  2,  0,  0],
                [ 0,  2,  2,  0,  4,  0,  0,  0,  0,  2,  0,  2,  0,  0,  2,  2]],
               dtype=np.uint8)
GIFT_RC = bytearray.fromhex(
    "0103070f1f3e3d3b372f1e3c3933270e1d3a352b162c"
    "18302102050b172e1c383123060d1b362d1a34291224"
    "081122040913260c1932250a152a14281020"
)
def unpack_bits(arr: np.ndarray, bitorder: Literal['big', 'little'] = 'little'):
    arr = np.array(arr, dtype=np.uint8)
    bits = np.unpackbits(arr, axis=-1, bitorder=bitorder)
    if bitorder == 'big':
        bit_offsets = [4, 5, 6, 7]
    else:
        bit_offsets = [0, 1, 2, 3]
    selector = np.stack([np.arange(16) * 8 + o for o in bit_offsets]).T.flatten()
    bits = bits[..., selector]
    return bits
def pack_bits(bits: np.ndarray, bitorder: Literal['big', 'little'] = 'little'):
    if bitorder == 'big':
        bit_offsets = [4, 5, 6, 7]
    else:
        bit_offsets = [0, 1, 2, 3]
    selector = np.stack([np.arange(16) * 8 + o for o in bit_offsets]).T.flatten()
    padded_bits = np.zeros((len(bits), 128), dtype=np.uint8)
    padded_bits[..., selector] = bits
    return np.packbits(padded_bits, axis=1, bitorder=bitorder)
def inverse_bit_perm(arr):
    bits = unpack_bits(arr, 'little')
    ip = bits[..., IP64]
    res = pack_bits(ip, 'little')
    return res
def bit_perm(arr):
    bits = unpack_bits(arr, 'little')
    pemuted = bits[..., P64]
    res = pack_bits(pemuted, 'little')
    return res
#!/usr/bin/env python3
import numpy as np
import argparse
from typing import Literal
P64 = np.array((0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3,
                4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,
                8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
                12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15))
IP64 = np.full_like(P64, -1)
IP64[P64] = np.arange(len(P64), dtype=P64.dtype)
IP64, P64 = P64, IP64
ddt = np.array([[16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
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
def unpack_bits(arr: np.ndarray, bitorder: Literal['big', 'little'] = 'little'):
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('output', default=None)
    args = parser.parse_args()
    res = []
    with open(args.filename, 'r') as f:
        for line in f:
            line = line.strip().replace(',', '').replace(' ', '')
            ints = [int(x, 16) for x in line[::-1]]
            res.append(ints)
    res = np.array(res, dtype=np.uint8)
    sbox_in = res[:-1]
    sbox_out = inverse_bit_perm(res[1:])
    assert sbox_in.shape == sbox_out.shape
    ddt_prob = np.log2(ddt[sbox_in, sbox_out] / 16).sum()
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    for inp, out in zip(sbox_in, sbox_out, strict=True):
        print(''.join(f'{x:x}' for x in inp)[::-1])
        print(''.join(f'{x:x}' for x in out)[::-1])
        print()
    if args.output:
        with open(args.output, 'w') as f:
            for inp, out in zip(sbox_in, sbox_out, strict=True):
                print(''.join(f'{x:x}' for x in inp)[::-1], file=f)
                print(''.join(f'{x:x}' for x in out)[::-1], file=f)
                print(file=f)
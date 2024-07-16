#!/usr/bin/env python3
import numpy as np
import argparse
from typing import Literal

P128 = np.array((0, 33, 66, 99, 96, 1, 34, 67, 64, 97, 2, 35, 32, 65, 98, 3, 4, 37, 70,
                103, 100, 5, 38, 71, 68, 101, 6, 39, 36, 69, 102, 7, 8, 41, 74, 107,
                104, 9, 42, 75, 72, 105, 10, 43, 40, 73, 106, 11, 12, 45, 78, 111, 108,
                13, 46, 79, 76, 109, 14, 47, 44, 77, 110, 15, 16, 49, 82, 115, 112, 17,
                50, 83, 80, 113, 18, 51, 48, 81, 114, 19, 20, 53, 86, 119, 116, 21, 54,
                87, 84, 117, 22, 55, 52, 85, 118, 23, 24, 57, 90, 123, 120, 25, 58, 91,
                88, 121, 26, 59, 56, 89, 122, 27, 28, 61, 94, 127, 124, 29, 62, 95, 92,
                125, 30, 63, 60, 93, 126, 31))


IP128 = np.full_like(P128, -1)
IP128[P128] = np.arange(len(P128), dtype=P128.dtype)

IP128, P128 = P128, IP128


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

    selector = np.stack([np.arange(32) * 8 + o for o in bit_offsets]).T.flatten()
    bits = bits[..., selector]
    return bits

def pack_bits(bits: np.ndarray, bitorder: Literal['big', 'little'] = 'little'):
    if bitorder == 'big':
        bit_offsets = [4, 5, 6, 7]
    else:
        bit_offsets = [0, 1, 2, 3]
    selector = np.stack([np.arange(32) * 8 + o for o in bit_offsets]).T.flatten()

    padded_bits = np.zeros((len(bits), 256), dtype=np.uint8)
    padded_bits[..., selector] = bits

    return np.packbits(padded_bits, axis=1, bitorder=bitorder)

def inverse_bit_perm(arr):
    bits = unpack_bits(arr, 'little')
    ip = bits[..., IP128]
    res = pack_bits(ip, 'little')
    return res

def bit_perm(arr):
    bits = unpack_bits(arr, 'little')
    pemuted = bits[..., P128]
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
            line = line.strip()
            if not line or line.startswith('#'):
                continue

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

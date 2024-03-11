#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import argparse
from typing import Literal
from midori128.util import DDT, RC, do_shift_rows, do_shift_rows_inv
from midori128.generate_perm import permutation
P128 = permutation()
def unpack_bits(cell):
    cellBin = [0 for _ in range(8)]
    for j in range(8):
        cellBin[j] = (cell >> j) & 0x01
    return cellBin
def pack_bits(cellBin):
    cell = 0;
    for j in range(8):
        cell = (cell << 1) | cellBin[7 - j];
    return cell
def unpack_bits_arr(A):
    B = []
    for a in A:
        B = B + unpack_bits(a)
    return B
def pack_bits_arr(A):
    B = []
    for i in range(len(A)//8):
        b = A[8*i:8*(i+1)]
        # print(b)
        B.append(pack_bits(b))
    B = np.asarray(B, dtype = np.uint8)
    return B
def bit_perm(arr_in):
    arr_out = []
    for A in arr_in:
        # print(A)
        B = np.asarray(unpack_bits_arr(A))
        # print(B)
        B = B[P128]
        B = pack_bits_arr(B)
        arr_out.append(B)
    arr_out = np.asarray(arr_out)
    return arr_out
def get_bytes(L):
    # print(len(L))
    I = []
    for i in range(len(L)):
        l = L[i]
        # print(i)
        # print(l)
        l = int(l, 16)
        I.append(l)
    return I
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
            line = line.strip().split(' ')
            line.reverse()
            # print(ints)
            ints = get_bytes(line)
            print(ints)
            res.append(ints)
    res = np.array(res, dtype=np.uint8)
    print(f'{res = }')
    sbox_in = bit_perm(res[:-1])
    print(sbox_in)
    sbox_out = bit_perm(res[1:])
    print(sbox_out)
    for i in range(sbox_out.shape[0]):
        sbox_out[i] = do_shift_rows_inv(sbox_out[i])
    print(sbox_out)
    assert sbox_in.shape == sbox_out.shape
    # for inp, out in zip(sbox_in, sbox_out, strict=True):
    #     print(''.join(f'{x:x}' for x in inp)[::-1])
    #     print(''.join(f'{x:x}' for x in out)[::-1])
    #     print()
    # if args.output:
    #     with open(args.output, 'w') as f:
    #         for inp, out in zip(sbox_in, sbox_out, strict=True):
    #             print(''.join(f'{x:x}' for x in inp)[::-1], file=f)
    #             print(''.join(f'{x:x}' for x in out)[::-1], file=f)
    #             print(file=f)
#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import argparse
from typing import Literal
from  .util import DDT, RC, do_shift_rows, do_shift_rows_inv, do_mix_columns
from  .util import nibble_to_byte, byte_to_nibble
from .generate_perm import permutation
P128 = permutation()
def unpack_bits(cell):
    cellBin = [0 for _ in range(4)]
    for j in range(4):
        cellBin[j] = (cell >> j) & 0x01
    return cellBin
def pack_bits(cellBin):
    cell = 0;
    for j in range(4):
        cell = (cell << 1) | cellBin[3 - j];
    return cell
def unpack_bits_arr(A):
    B = []
    for a in A:
        B = B + unpack_bits(a)
    return B
def pack_bits_arr(A):
    B = []
    for i in range(len(A)//4):
        b = A[4*i:4*(i+1)]
        # print(b)
        B.append(pack_bits(b))
    B = np.asarray(B, dtype = np.uint8)
    return B
def bit_perm(arr_in):
    arr_out = []
    for A in arr_in:
        # print(A)
        B = np.asarray(unpack_bits_arr(A))
        B = B[P128]
        B = pack_bits_arr(B)
        # print(B)
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
            line = ''.join(line)
            ints = [int(x, 16) for x in line]
            # ints.reverse()
            res.append(ints)
    res = np.array(res, dtype=np.uint8)
    print(res)
    sbox_in = res.copy()[:-1]
    sbox_out = res.copy()[1:]
    for i in range(sbox_out.shape[0]):
        sbox_out[i] = byte_to_nibble(do_mix_columns(nibble_to_byte(sbox_out[i])))
        sbox_out[i] = byte_to_nibble(do_shift_rows_inv(nibble_to_byte(sbox_out[i])))
    sbox_in_or  = sbox_in.copy()
    sbox_out_or = sbox_out.copy()
    sbox_in = bit_perm(sbox_in)
    sbox_out = bit_perm(sbox_out)
    print(sbox_in)
    print(sbox_out)
    assert sbox_in.shape == sbox_out.shape
    for inp, out in zip(sbox_in, sbox_out, strict=True):
        print(DDT[inp, out])
    ddt_prob = np.log2(DDT[sbox_in, sbox_out] / 16).sum()
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    for inp, out in zip(sbox_in, sbox_out, strict=True):
        print(''.join(f'{x:x}' for x in inp)[::])
        print(''.join(f'{x:x}' for x in out)[::])
        print()
    char = []
    for inp, out in zip(sbox_in, sbox_out, strict=True):
        s = []
        s.append(''.join(f'{x:x}' for x in inp)[::])
        s.append(''.join(f'{x:x}' for x in out)[::])
        s = tuple(s)
        char.append(s)
    print('char = ', tuple(char))
    char = []
    for inp, out in zip(sbox_in_or, sbox_out_or, strict=True):
        s = []
        s.append(''.join(f'{x:x}' for x in inp)[::])
        s.append(''.join(f'{x:x}' for x in out)[::])
        s = tuple(s)
        char.append(s)
    print('char1 = ', tuple(char))
    # if args.output:
    #     with open(args.output, 'w') as f:
    #         for inp, out in zip(sbox_in, sbox_out, strict=True):
    #             print(''.join(f'{x:x}' for x in inp)[::-1], file=f)
    #             print(''.join(f'{x:x}' for x in out)[::-1], file=f)
    #             print(file=f)
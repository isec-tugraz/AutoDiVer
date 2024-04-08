#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import argparse
from typing import Literal
from  .util import DDT, RC, perm_nibble_inv, perm_nibble_16, perm_nibble_16_inv, perm_nibble
from  .util import get_round_in_out
def print_state(S):
    for s in S:
        print(hex(s)[2:], end = "")
    print("")
def check_L_R(L, R):
    print(" " )
    for i in range(len(L) - 1):
        a = L[i].copy()
        b = R[i+1].copy()
        print_state(a)
        print_state(b)
        # a = perm_nibble_16(a)
        # print_state(a)
        assert np.all(a == b)
def get_sbox_in_out(inds):
    inds_d = inds.shape
    L = np.empty((inds_d[0], 16), dtype=np.uint8)
    R = np.empty((inds_d[0], 16), dtype=np.uint8)
    for i in range(inds_d[0]):
        ind = inds[i]
        for j in range(16):
            L[i][j] = ind[2*j]
            R[i][j] = ind[2*j + 1]
        print_state(inds[i])
        print_state(L[i])
        print_state(R[i])
    check_L_R(L, R)
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
            ints = [int(x, 16) for x in line]
            ints.reverse()
            res.append(ints)
    inds = np.array(res, dtype=np.uint8)
    print(inds)
    sbox_in, sbox_out = get_sbox_in_out(inds)
    # for i in range(sbox_out.shape[0]):
    #     sbox_out[i] = byte_to_nibble(do_mix_columns(nibble_to_byte(sbox_out[i])))
    #     sbox_out[i] = byte_to_nibble(do_shift_rows_inv(nibble_to_byte(sbox_out[i])))
    # sbox_in_or  = sbox_in.copy()
    # sbox_out_or = sbox_out.copy()
    # sbox_in = bit_perm(sbox_in)
    # sbox_out = bit_perm(sbox_out)
    # print(sbox_in)
    # print(sbox_out)
    # assert sbox_in.shape == sbox_out.shape
    # for inp, out in zip(sbox_in, sbox_out, strict=True):
    #     print(DDT[inp, out])
    # ddt_prob = np.log2(DDT[sbox_in, sbox_out] / 16).sum()
    # print(f"ddt probability: 2**{ddt_prob:.1f}")
    # for inp, out in zip(sbox_in, sbox_out, strict=True):
    #     print(''.join(f'{x:x}' for x in inp)[::])
    #     print(''.join(f'{x:x}' for x in out)[::])
    #     print()
    # char = []
    # for inp, out in zip(sbox_in, sbox_out, strict=True):
    #     s = []
    #     s.append(''.join(f'{x:x}' for x in inp)[::])
    #     s.append(''.join(f'{x:x}' for x in out)[::])
    #     s = tuple(s)
    #     char.append(s)
    # print('char = ', tuple(char))
    # char = []
    # for inp, out in zip(sbox_in_or, sbox_out_or, strict=True):
    #     s = []
    #     s.append(''.join(f'{x:x}' for x in inp)[::])
    #     s.append(''.join(f'{x:x}' for x in out)[::])
    #     s = tuple(s)
    #     char.append(s)
    # print('char1 = ', tuple(char))
    # # if args.output:
    # #     with open(args.output, 'w') as f:
    # #         for inp, out in zip(sbox_in, sbox_out, strict=True):
    # #             print(''.join(f'{x:x}' for x in inp)[::-1], file=f)
    # #             print(''.join(f'{x:x}' for x in out)[::-1], file=f)
    # #             print(file=f)
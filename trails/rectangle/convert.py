#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import argparse
from typing import Literal

ddt = np.array([
[16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 2, 0, 0, 4, 2, 0, 0, 0, 2, 0, 0, 4, 2],
[0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 0, 2, 4, 0, 2],
[0, 0, 0, 2, 0, 0, 2, 0, 2, 4, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4],
[0, 2, 0, 0, 4, 2, 0, 0, 4, 2, 0, 0, 0, 2, 0, 0],
[0, 2, 4, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2],
[0, 0, 4, 0, 2, 2, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2],
[0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
[0, 2, 0, 0, 0, 2, 4, 0, 0, 2, 0, 0, 0, 2, 4, 0],
[0, 0, 0, 0, 0, 4, 2, 2, 2, 0, 2, 0, 2, 0, 0, 2],
[0, 4, 0, 2, 0, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 0],
[0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4, 0, 0, 2, 4, 0],
[0, 0, 4, 2, 2, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
[0, 2, 4, 2, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0]],
dtype=np.uint8)

"""
Bit State is a two dimensional nparray of size 4*16
"""
def bitStateToNibbleState(bitstate):
    # Convert each column of 4 bits to an integer
    nibbles = [0 for i in range(16)]
    for col in range(16):
        nibble = 0
        for row in range(4):
            b = bitstate[16*row + col] & 0x1
            nibble = nibble | (b << row)
        nibbles[col] = nibble
    return nibbles

def get_sbox_in_out(inds):
    inds_d = inds.shape
    sbox_in  = np.empty((inds_d[0]//2, 16), dtype=np.uint8)
    sbox_out = np.empty((inds_d[0]//2, 16), dtype=np.uint8)
    for i in range(inds_d[0]//2):
        sbox_in[i] = inds[2*i].copy()
        sbox_out[i] = inds[2*i+1].copy()
    return sbox_in, sbox_out

if __name__ == '__main__':
    BITSTATE = True
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
            if BITSTATE == True:
                bits = [int(x, 16) for x in line]
                nibbles = bitStateToNibbleState(bits)
            else:
                nibbles = [int(x, 16) for x in line]
                nibbles.reverse()
            res.append(nibbles)
    inds = np.array(res, dtype=np.uint8)
    print(inds)
    sbox_in, sbox_out = get_sbox_in_out(inds)

    assert sbox_in.shape == sbox_out.shape

    ddt_prob = np.log2(ddt[sbox_in, sbox_out] / 16).sum()
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    for inp, out in zip(sbox_in, sbox_out, strict=True):
        print(''.join(f'{x:x}' for x in inp))
        print(''.join(f'{x:x}' for x in out))
        print()


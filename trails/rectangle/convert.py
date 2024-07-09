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

def get_sbox_in_out(inds):
    inds_d = inds.shape
    sbox_in  = np.empty((inds_d[0]//2, 16), dtype=np.uint8)
    sbox_out = np.empty((inds_d[0]//2, 16), dtype=np.uint8)
    for i in range(inds_d[0]//2):
        sbox_in[i] = inds[2*i].copy()
        sbox_out[i] = inds[2*i+1].copy()
    return sbox_in, sbox_out

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

    assert sbox_in.shape == sbox_out.shape

    ddt_prob = np.log2(ddt[sbox_in, sbox_out] / 16).sum()
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    for inp, out in zip(sbox_in, sbox_out, strict=True):
        print(''.join(f'{x:x}' for x in inp))
        print(''.join(f'{x:x}' for x in out))
        print()


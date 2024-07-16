#!/usr/bin/env python3
"""
Toy example for present with 8 active s-boxes in 2 rounds
"""
from pathlib import Path
import numpy as np

from present_util import bit_perm

ddt = np.array([[16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  4,  0,  0,  0,  4,  0,  4,  0,  0,  0,  4,  0,  0],
                [ 0,  0,  0,  2,  0,  4,  2,  0,  0,  0,  2,  0,  2,  2,  2,  0],
                [ 0,  2,  0,  2,  2,  0,  4,  2,  0,  0,  2,  2,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  4,  2,  2,  0,  2,  2,  0,  2,  0,  2,  0],
                [ 0,  2,  0,  0,  2,  0,  0,  0,  0,  2,  2,  2,  4,  2,  0,  0],
                [ 0,  0,  2,  0,  0,  0,  2,  0,  2,  0,  0,  4,  2,  0,  0,  4],
                [ 0,  4,  2,  0,  0,  0,  2,  0,  2,  0,  0,  0,  2,  0,  0,  4],
                [ 0,  0,  0,  2,  0,  0,  0,  2,  0,  2,  0,  4,  0,  2,  0,  4],
                [ 0,  0,  2,  0,  4,  0,  2,  0,  2,  0,  0,  0,  2,  0,  4,  0],
                [ 0,  0,  2,  2,  0,  4,  0,  0,  2,  0,  2,  0,  0,  2,  2,  0],
                [ 0,  2,  0,  0,  2,  0,  0,  0,  4,  2,  2,  2,  0,  2,  0,  0],
                [ 0,  0,  2,  0,  0,  4,  0,  2,  2,  2,  2,  0,  0,  0,  2,  0],
                [ 0,  2,  4,  2,  2,  0,  0,  2,  0,  0,  2,  2,  0,  0,  0,  0],
                [ 0,  0,  2,  2,  0,  0,  2,  2,  2,  2,  0,  0,  2,  2,  0,  0],
                [ 0,  4,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  4]])


if __name__ == '__main__':
    nrounds = 2
    sbox_in = np.zeros((nrounds, 16), dtype=np.uint8)
    sbox_out = np.zeros((nrounds, 16), dtype=np.uint8)

    sbox_in[0, [15, 14, 13, 12]] = [1, 2, 4, 8]
    sbox_out[0, [15, 14, 13, 12]] = [3, 5, 5, 11]

    sbox_in[1] = bit_perm([sbox_out[0]])[0]
    sbox_out[1] = np.argmax(ddt[sbox_in[1], :], axis=1)


    print("".join(f"{x:x}" for x in reversed(sbox_in[0])))
    print("".join(f"{x:x}" for x in reversed(sbox_out[0])))
    print("".join(f"{x:x}" for x in reversed(sbox_in[1])))
    print("".join(f"{x:x}" for x in reversed(sbox_out[1])))

    # sbox_in[3, [0, 3]] = 4

    prob = np.log2(ddt[sbox_in, sbox_out] / 16).sum()
    print(f"probability: 2^{prob:.1f}")

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, sbox_in=sbox_in, sbox_out=sbox_out)
    # numrounds = len(sbox_in)

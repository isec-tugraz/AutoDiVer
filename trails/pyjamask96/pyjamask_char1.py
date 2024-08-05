from __future__ import annotations

from pathlib import Path
import numpy as np

from typing import Any

if __name__ == '__main__':

    sbox_in = np.array([
        [0x00000000, 0x00000000, 0x00a04e67],
        [0x00000000, 0xa900010a, 0x00000000],
        [0x2040b886, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x04010c62],
        [0x00000000, 0x0a3a0841, 0x00000000],
    ]).astype(np.uint32)

    sbox_out = np.zeros([5, 3]).astype(np.uint32)

    for i in range(len(sbox_in)):
        sbox_out[i] = np.roll(sbox_in[i], -1, axis=0)

    hex_array = np.vectorize(lambda x: f"{x:08x}")(sbox_in[:-1])
    print('\n'.join([' '.join(row) for row in hex_array]))
    print("\n")
    hex_array = np.vectorize(lambda x: f"{x:08x}")(sbox_out)
    print('\n'.join([' '.join(row) for row in hex_array]))

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, sbox_in=sbox_in, sbox_out=sbox_out)

    numrounds = len(sbox_out)
    for i in range(1, numrounds):
        dst_file = script_file.with_name(script_file.stem + f'_r{i}.npz')
        print(f'Writing to {dst_file}')
        np.savez(dst_file, sbox_in=sbox_in[:i], sbox_out=sbox_out[:i])

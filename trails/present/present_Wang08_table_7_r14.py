#!/usr/bin/env python3
"""
@inproceedings{africacrypt/Wang08,
    author = {Meiqin Wang},
    title = {Differential Cryptanalysis of Reduced-Round {PRESENT}},
    booktitle = {{AFRICACRYPT} 2008},
    series = {LNCS},
    volume = {5023},
    pages = {40--49},
    publisher = {Springer},
    year = {2008},
    doi = {10.1007/978-3-540-68164-9_4},
    biburl = {https://dblp.org/rec/conf/africacrypt/Wang08.bib},
}

Table 7: The 14-round Differential of PRESENT
"""
from pathlib import Path
import numpy as np

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
    nrounds = 14
    sbox_in = np.zeros((nrounds, 16), dtype=np.uint8)
    sbox_out = np.zeros((nrounds, 16), dtype=np.uint8)

    sbox_in[0, [2, 14]] = 7
    sbox_out[0, [2, 14]] = 1

    sbox_in[1, [0, 3]] = 4
    sbox_out[1, [0, 3]] = 5

    sbox_in[2, [0, 8]] = 9
    sbox_out[2, [0, 8]] = 4

    sbox_in[3, [8, 10]] = 1
    sbox_out[3, [8, 10]] = 9

    sbox_in[4, [2, 14]] = 5
    sbox_out[4, [2, 14]] = 1

    sbox_in[5, [0, 3]] = 4
    sbox_out[5, [0, 3]] = 5

    sbox_in[6, [0, 8]] = 9
    sbox_out[6, [0, 8]] = 4

    sbox_in[7, [8, 10]] = 1
    sbox_out[7, [8, 10]] = 9

    sbox_in[8, [2, 14]] = 5
    sbox_out[8, [2, 14]] = 1

    sbox_in[9, [0, 3]] = 4
    sbox_out[9, [0, 3]] = 5

    sbox_in[10, [0, 8]] = 9
    sbox_out[10, [0, 8]] = 4

    sbox_in[11, [8, 10]] = 1
    sbox_out[11, [8, 10]] = 9

    sbox_in[12, [2, 14]] = 5
    sbox_out[12, [2, 14]] = 1

    sbox_in[13, [0, 3]] = 4
    sbox_out[13, [0, 3]] = 5

    # sbox_in[14, [0, 8]] = 0
    print(np.where(ddt[sbox_in, sbox_out] == 0))

    prob = np.log2(ddt[sbox_in, sbox_out] / 16).sum()
    print(f"probability: 2^{prob:.1f}")

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, sbox_in=sbox_in, sbox_out=sbox_out)

    for i in [8, 10, 12]:
        dst_file = script_file.with_name(script_file.stem.replace('r14', f'r{i}.npz'))
        print(f'Writing to {dst_file}')
        np.savez(dst_file, sbox_in=sbox_in[:i], sbox_out=sbox_out[:i])

    for i in [8, 10, 12]:
        dst_file = script_file.with_name(script_file.stem.replace('r14', f'r{i}_last.npz'))
        print(f'Writing to {dst_file}')
        np.savez(dst_file, sbox_in=sbox_in[-i:], sbox_out=sbox_out[-i:])

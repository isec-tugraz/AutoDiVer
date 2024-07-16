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

Table 5 4-round Iterative Differential of PRESENT
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
    nrounds = 4
    sbox_in = np.zeros((nrounds, 16), dtype=np.uint8)
    sbox_out = np.zeros((nrounds, 16), dtype=np.uint8)

    sbox_in[0, [0, 3]] = 4
    sbox_out[0, [0, 3]] = 5

    sbox_in[1, [0, 8]] = 9
    sbox_out[1, [0, 8]] = 4

    sbox_in[2, [8, 10]] = 1
    sbox_out[2, [8, 10]] = 9

    sbox_in[3, [2, 14]] = 5
    sbox_out[3, [2, 14]] = 1

    # sbox_in[3, [0, 3]] = 4

    prob = np.log2(ddt[sbox_in, sbox_out] / 16).sum()
    print(f"probability: 2^{prob:.1f}")

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, sbox_in=sbox_in, sbox_out=sbox_out)
    # numrounds = len(sbox_in)

    for iters in range(2, 5):
        dst_file = script_file.with_name(script_file.stem.replace('r4', f'r{4*iters}').replace('iter', f'iter{iters}'))
        print(f'Writing to {dst_file} with {4*iters} rounds')

        sbox_in_rep = np.tile(sbox_in, (iters, 1))
        print(sbox_in_rep)
        sbox_out_rep = np.tile(sbox_out, (iters, 1))
        np.savez(dst_file, sbox_in=sbox_in_rep, sbox_out=sbox_out_rep)

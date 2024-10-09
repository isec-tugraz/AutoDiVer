#!/usr/bin/env python3
"""
Cahracteristic from Table 7 of

@inproceedings{fse/AbedLLW14,
    author = {Farzaneh Abed and Eik List and Stefan Lucks and Jakob Wenzel},
    title = {Differential Cryptanalysis of Round-Reduced {Simon} and {Speck}},
    booktitle = {{FSE} 2014},
    series = {LNCS},
    volume = {8540},
    pages = {525--545},
    publisher = {Springer},
    year = {2014},
    doi = {10.1007/978-3-662-46706-0_27},
    biburl = {https://dblp.org/rec/conf/fse/AbedLLW14.bib},
    xeditor = {Carlos Cid and Christian Rechberger},
}

probability: 2^-41
probability_acc: 2^{-40.55}
"""
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    diff_indices = [
        ([0,8,9,11,19,22],  [0,3,14,16,19]),
        ([1,11,12,19],      [1,3,6,11,17,22]),
        ([1,4,6,22],        [9,14,20,22]),
        ([9,17,23],         [1,9,12]),
        ([12,15],           [4]),
        ([7],               []),
        ([23],              [23]),
        ([15,23],           [2,15,23]),
        ([2,7,23],          [5,7,18,23]),
        ([5,7,15],          [2,5,7,8,10,15,21]),
        ([2,5,8,10,15,23],  [0,2,11,13,15,18,23]),
    ]

    round_in = np.array([
        [sum(1 << j for j in left), sum(1 << j for j in right)] for left, right in diff_indices
    ], dtype=np.uint64)
    wordsize = 24

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

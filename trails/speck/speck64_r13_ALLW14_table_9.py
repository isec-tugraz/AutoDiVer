#!/usr/bin/env python3
"""
Characteristic from Table 9 of

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

probability: 2^-59
probability_acc: 2^{-58.9}
"""
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    diff_indices = [
        ([5,21,24,27,30],   [8,13,19,29]),
        ([8,16,22],         [0,8,11]),
        ([11,14],           [3]),
        ([6],               []),
        ([30],              [30]),
        ([22,30],           [1,22,30]),
        ([1,14,30],         [4,14,25,30]),
        ([4,6,7,14,22,30],  [1,4,6,14,17,22,28,30]),
        ([1,4,7,17,31],     [9,20,25]),
        ([20,23,28,31],     [12,20,31]),
        ([15,23,31],        [2,31]),
        ([2,7,15,23,31],    [5,7,15,23,31]),
        ([5,26],            [2,5,8,10,18]),
        ([2,5,8,10,29],     [2,10,11,13,21,29]),
    ]

    round_in = np.array([
        [sum(1 << j for j in left), sum(1 << j for j in right)] for left, right in diff_indices
    ], dtype=np.uint64)
    wordsize = 32

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

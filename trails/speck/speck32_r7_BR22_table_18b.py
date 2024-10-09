#!/usr/bin/env python3
"""
Characteristic from Table 18 (right) of

@article{iacr/BeyneR22,
    author = {Tim Beyne and Vincent Rijmen},
    title = {Differential Cryptanalysis in the Fixed-Key Model},
    journal = {{IACR} Cryptol. ePrint Arch.},
    pages = {837},
    year = {2022},
    url = {https://eprint.iacr.org/2022/837},
    biburl = {https://dblp.org/rec/journals/iacr/BeyneR22.bib},
}
probability: 2^-18
originally from

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
"""
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    round_in = np.array([
        [0x0a60, 0x4205],
        [0x0211, 0x0a04],
        [0x2800, 0x0010],
        [0x0040, 0x0000],
        [0x8000, 0x8000],
        [0x8100, 0x8102],
        [0x8000, 0x840a],
        [0x850a, 0x9520],
    ], dtype=np.uint64)
    wordsize = 16

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

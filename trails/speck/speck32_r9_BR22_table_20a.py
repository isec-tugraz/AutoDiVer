#!/usr/bin/env python3
"""
Characteristic from Table 20 (left) of

@article{iacr/BeyneR22,
    author = {Tim Beyne and Vincent Rijmen},
    title = {Differential Cryptanalysis in the Fixed-Key Model},
    journal = {{IACR} Cryptol. ePrint Arch.},
    pages = {837},
    year = {2022},
    url = {https://eprint.iacr.org/2022/837},
    biburl = {https://dblp.org/rec/journals/iacr/BeyneR22.bib},
}
probability: 2^-30
originally from Table 6 (left) of

@inproceedings{fse/Biryukov0V14,
    author = {Alex Biryukov and Arnab Roy and Vesselin Velichkov},
    title = {Differential Analysis of Block Ciphers {SIMON} and {SPECK}},
    booktitle = {{FSE} 2014},
    series = {LNCS},
    volume = {8540},
    pages = {546--570},
    publisher = {Springer},
    year = {2014},
    doi = {10.1007/978-3-662-46706-0_28},
    biburl = {https://dblp.org/rec/conf/fse/Biryukov0V14.bib},
    xeditor = {Carlos Cid and Christian Rechberger},
}
"""
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    round_in = np.array([
        [0x8054, 0xa900],
        [0x0000, 0xa402],
        [0xa402, 0x3408],
        [0x50c0, 0x80e0],
        [0x0181, 0x0203],
        [0x000c, 0x0800],
        [0x2000, 0x0000],
        [0x0040, 0x0040],
        [0x8040, 0x8140],
        [0x0040, 0x0542],
    ], dtype=np.uint64)
    wordsize = 16

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

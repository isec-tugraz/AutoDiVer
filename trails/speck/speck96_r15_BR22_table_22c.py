#!/usr/bin/env python3
"""
Characteristic from Table 22 (right) of

@article{iacr/BeyneR22,
    author = {Tim Beyne and Vincent Rijmen},
    title = {Differential Cryptanalysis in the Fixed-Key Model},
    journal = {{IACR} Cryptol. ePrint Arch.},
    pages = {837},
    year = {2022},
    url = {https://eprint.iacr.org/2022/837},
    biburl = {https://dblp.org/rec/journals/iacr/BeyneR22.bib},
}

originally from

@inproceedings{acisp/SongHY16,
    author = {Ling Song and Zhangjie Huang and Qianqian Yang},
    title = {Automatic Differential Analysis of {ARX} Block Ciphers with Application to {SPECK} and {LEA}},
    booktitle = {{ACISP} 2016},
    series = {LNCS},
    volume = {9723},
    pages = {379--394},
    publisher = {Springer},
    year = {2016},
    doi = {10.1007/978-3-319-40367-0_24},
    biburl = {https://dblp.org/rec/conf/acisp/SongHY16.bib},
    xeditor = {Joseph K. Liu and Ron Steinfeld},
}

probability: 2^-81
"""
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    round_in = np.array([
        [0x082020000000, 0x000120200000],
        [0x000900000000, 0x000001000000],
        [0x000008000000, 0x000000000000],
        [0x000000080000, 0x000000080000],
        [0x000000080800, 0x000000480800],
        [0x000000480008, 0x000002084008],
        [0x0800fe080808, 0x0800ee4a0848],
        [0x000772400040, 0x400000104200],
        [0x000000820200, 0x000000001202],
        [0x000000009000, 0x000000000010],
        [0x000000000080, 0x000000000000],
        [0x800000000000, 0x800000000000],
        [0x808000000000, 0x808000000004],
        [0x800080000004, 0x840080000020],
        [0x808080800020, 0xa08480800124],
        [0x800400008124, 0x842004008801],
    ], dtype=np.uint64)
    wordsize = 48

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

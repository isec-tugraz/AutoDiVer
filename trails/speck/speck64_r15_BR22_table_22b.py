#!/usr/bin/env python3
"""
Characteristic from Table 22 (middle) of

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

probability: 2^-62
"""
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    round_in = np.array([
        [0x04092400, 0x20040104],
        [0x20000820, 0x20200001],
        [0x00000009, 0x01000000],
        [0x08000000, 0x00000000],
        [0x00080000, 0x00080000],
        [0x00080800, 0x00480800],
        [0x00480008, 0x02084008],
        [0x06080808, 0x164a0848],
        [0xf2400040, 0x40104200],
        [0x00820200, 0x00001202],
        [0x00009000, 0x00000010],
        [0x00000080, 0x00000000],
        [0x80000000, 0x80000000],
        [0x80800000, 0x80800004],
        [0x80008004, 0x84008020],
        [0x808080a0, 0xa08481a4],
    ], dtype=np.uint64)
    wordsize = 32

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

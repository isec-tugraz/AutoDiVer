#!/usr/bin/env python3
"""
Characteristic from Table 23 (bottom left) of

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

probability: 2^-128
"""
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    round_in = np.array([
        [0x0124000400000000, 0x0801042004000000],
        [0x0800202000000000, 0x4808012020000000],
        [0x4800010000000000, 0x0840080100000002],
        [0x0808080000000002, 0x4a08480800000012],
        [0x4400400000000012, 0x1442004000000080],
        [0x2202000000000080, 0x8012020000000480],
        [0x0010000000000480, 0x0080100000002084],
        [0x8080000000006080, 0x84808000000164a0],
        [0x0400000000032400, 0x2004000000080104],
        [0x2000000000080020, 0x2020000000480801],
        [0x0000000000480001, 0x0100000002084008],
        [0x000000000e080808, 0x080000001e4a0848],
        [0x00000000f2400040, 0x4000000000104200],
        [0x0000000000820200, 0x0000000000001202],
        [0x0000000000009000, 0x0000000000000010],
        [0x0000000000000080, 0x0000000000000000],
        [0x8000000000000000, 0x8000000000000000],
        [0x8080000000000000, 0x8080000000000004],
        [0x8000800000000004, 0x8400800000000020],
        [0x8080808000000020, 0xa084808000000124],
        [0x8004000080000124, 0x8420040080000801],
    ], dtype=np.uint64)
    wordsize = 64

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

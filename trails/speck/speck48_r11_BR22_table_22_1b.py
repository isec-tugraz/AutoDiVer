#!/usr/bin/env python3
"""
Cahracteristic from Table 22 (right of leftmost subtable) of

@article{iacr/BeyneR22,
    author = {Tim Beyne and Vincent Rijmen},
    title = {Differential Cryptanalysis in the Fixed-Key Model},
    journal = {{IACR} Cryptol. ePrint Arch.},
    pages = {837},
    year = {2022},
    url = {https://eprint.iacr.org/2022/837},
    biburl = {https://dblp.org/rec/journals/iacr/BeyneR22.bib},
}
probability: 2^-47
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
"""
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    round_in = np.array([
        [0x504200, 0x004240],
        [0x001202, 0x020002],
        [0x000010, 0x100000],
        [0x000000, 0x800000],
        [0x800000, 0x800004],
        [0x808004, 0x808020],
        [0x8400a0, 0x8001a4],
        [0xe08da4, 0xe08080],
        [0x042007, 0x002400],
        [0x012020, 0x000020],
        [0x200100, 0x200000],
        [0x202001, 0x202000],
    ], dtype=np.uint64)
    wordsize = 24

    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, round_in=round_in, wordsize=wordsize)

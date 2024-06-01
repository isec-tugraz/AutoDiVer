#!/usr/bin/env python3
"""
MILP-Based Differential Cryptanalysis on Round-Reduced Midori64
Hongluan Zhao, Guoyong Han, Letian Wang and Wen Wang
https://doi.org/10.1109/ACCESS.2020.2995795
Table 5
Probability: 2^-52
"""
from pathlib import Path
import numpy as np
def from_hex(s: str):
    return np.array([int(x, 16) for x in s], dtype=np.uint8)
def state2tex(S):
    res = ""
    for row in range(4):
        for col in range(4):
            if S[row][col]:
                res += r"\Cell{ss" + str(row) + str(col) + "}{" + hex(S[row][col])[2:] + "}"
    return res
if __name__ == '__main__':
    alpha = 1
    assert alpha in [0x1, 0x4, 0x9, 0xc]
    char = np.array([
        (from_hex("1000000000100000"), from_hex("2000000000200000")),
        (from_hex("2200000000000000"), from_hex("4100000000000000")),
        (from_hex("0444111000000000"), from_hex("0222222000000000")),
        (from_hex("2202020202022202"), from_hex("4101040101011104")),
        (from_hex("0400001100011100"), from_hex("0200002200022200")),
    ])
    sbox_in = char[:, 0, :].reshape(-1, 4, 4).swapaxes(-1, -2)
    sbox_out = char[:, 1, :].reshape(-1, 4, 4).swapaxes(-1, -2)
    for r, (si, so) in enumerate(zip(sbox_in, sbox_out)):
        if r:
            print(r"\CipherLine{}" + "{" + state2tex(si) + "}")
        else:
            print(r"\CipherInit{}" + "{" + state2tex(si) + "}")
        print(r"\CipherStep{\cipher{SB}}{}" + "{" + state2tex(so) + "}")
        if r == len(sbox_in) - 1:
            pass # ?
        else:
            print(r"\CipherStep{\cipher{SR\\MC\\AK}}{}" + "{" + state2tex(so) + "}") # TODO key
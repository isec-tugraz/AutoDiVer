#!/usr/bin/env python3
"""
18-round differential characteristic for WARP.

# MILP Based Differential Attack on Round Reduced WARP
# Manoj Kumar and Tarun Yadav
#
# https://doi.org/10.1007/978-3-030-95085-9_3
# Table 9. 18-round differential characteristics
# p = 2^-122

The paper lists the trail in WARP's LBlock-like equivalent view, so we need to
convert according to Figure 11 in the WARP specification.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PERM = np.asarray([31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10, 15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26])

# WARP spec Fig. 11b: the nibble shuffle relating the LBlock and classical states
PIR = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5])
PIL = np.array([5, 4, 6, 0, 3, 7, 2, 1, 13, 12, 14, 8, 11, 15, 10, 9])

# round input differences X^r = (x^r_31, ..., x^r_0)
ROUNDS_IN = [
    "000af000faf000000a0000005f500050",
    "00a00500a0000000af000000f0000000",
    "00005f0000000000f000000000000a00",
    "0000f0000000000a000000000000af00",
    "00000000000000a0000000000000f000",
    "000000000f00000000000000000a0000",
    "0000000ff0000000000a000000a00000",
    "000000fa00000a0000a00f00000a0000",
    "0a0000a00000aa000000f00000a00000",
    "a0000f0a0000a000000a00000a000000",
    "0000f0a00000000a00a0000aaa050000",
    "00000a000f0000a00f0d00aaa0500a0a",
    "000aaa0af000000affd00aa00d00ada0",
    "00aaa0a0000005aaf000a00ad000df00",
    "00a00d00000a50aa000a00a0000af000",
    "0000d0000aa000a000ad000005a00000",
    "00000000a00000000dd000005000000a",
    "0000050000000000d00000000a0000ad",
    "00005000000a00070000000da7000dd0",
]


if __name__ == '__main__':
    lblock = np.array([[int(c, 16) for c in row[::-1]] for row in ROUNDS_IN], dtype=np.uint8)

    classical = np.empty_like(lblock)
    classical[:, 0::2] = lblock[:, 0::2][:, PIL]
    classical[:, 1::2] = lblock[:, 1::2][:, PIR]

    num_rounds = len(classical) - 1
    sbox_in = classical[:num_rounds, 0::2].copy()
    sbox_out = np.zeros((num_rounds, 16), dtype=np.uint8)
    for r in range(num_rounds):
        before_perm = classical[r + 1][PERM]
        assert np.array_equal(before_perm[0::2], classical[r, 0::2]), f"Feistel mismatch at round {r}"
        sbox_out[r] = before_perm[1::2] ^ classical[r, 1::2]

    dst_file = Path(__file__).with_suffix('.npz')
    print(f'Writing to {dst_file} (num_rounds={num_rounds}, probability=2^{round(log2_p)})')
    np.savez(dst_file, sbox_in=sbox_in, sbox_out=sbox_out)

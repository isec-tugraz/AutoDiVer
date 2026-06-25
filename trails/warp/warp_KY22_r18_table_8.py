#!/usr/bin/env python3
"""
18-round differential characteristic for WARP.

# MILP Based Differential Attack on Round Reduced WARP
# Manoj Kumar and Tarun Yadav
#
# https://doi.org/10.1007/978-3-030-95085-9_3
# Table 8. 18-round differential characteristics (extended from 17-round)
# p = 2^-122

The paper describes WARP in an LBlock-like equivalent view (Fig. 4 / Algorithm 1):
the 32-nibble round input difference is X^r = (x^r_31, ..., x^r_0); the even and odd
nibbles form X0 = (x_0, x_2, ..., x_30) and X1 = (x_1, x_3, ..., x_31). One round is

    Y    = S(X0^r)                       # s-box layer (X0 are the s-box inputs)
    U    = N_P(Y (+) K)                  # paper nibble permutation N_P (Table 3)
    V    = X1^r <<< 24 bits              # rotate the odd nibbles left by 6 nibbles
    X0^{r+1} = U (+) V
    X1^{r+1} = X0^r                      # s-box inputs pass through to the next X1

so the per-round s-box output difference is recovered from two consecutive round
inputs as

    sbox_in[r]  = X0^r
    sbox_out[r] = N_P^{-1}( X0^{r+1} (+) (X1^r <<< 6 nibbles) )

with the consistency condition X1^{r+1} == X0^r.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from autodiver.warp128.util import DDT

# paper nibble permutation N_P (Table 3): N_P(i) is the destination of nibble i
N_P = np.array([3, 7, 6, 4, 1, 0, 2, 5, 11, 15, 14, 12, 9, 8, 10, 13])

# round input differences X^r = (x^r_31, ..., x^r_0), one row per round (19 -> 18 rounds)
ROUNDS_IN = [
    "0007a000fa7000000a000000d5f000d0",
    "00700d00a0000000aa00000050000000",
    "0000d50000000000a000000000000a00",
    "000050000000000a000000000000aa00",
    "00000000000000a0000000000000a000",
    "000000000a00000000000000000a0000",
    "0000000aa0000000000a000000a00000",
    "000000aa00000a0000a00a00000a0000",
    "0a0000a00000af000000a00000a00000",
    "a0000f0a0000f000000a00000a000000",
    "0000f0a00000000a00a0000aaf0f0000",
    "00000a000a0000a00f0500aaf0f00a0a",
    "000aaf0aa000000afa500aa00500ada0",
    "00aaf0a000000aaaa000a00a5000df00",
    "00a00500000fa0aa000a00a0000af000",
    "000050000af000a000a500000aa00000",
    "00000000a000000005500000a000000a",
    "00000a0000000000500000000a0000a5",
    "0000a000000a000f0000000fa7000550",
]


PIR = [int(x) for x in "6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5".split()]
PIL = [int(x) for x in "5 4 6 0 3 7 2 1 13 12 14 8 11 15 10 9".split()]
POL = [int(x) for x in "3 7 6 4 1 0 2 5 11 15 14 12 9 8 10 13".split()]


def nibble_perm_inverse(nibbles: np.ndarray) -> np.ndarray:
    """Apply the inverse of the paper nibble permutation N_P to 16 nibbles."""
    out = np.empty_like(nibbles)
    out[:] = nibbles[N_P]
    return out


if __name__ == '__main__':
    # parse x^r_31..x^r_0 into state[i] = x_i, then split into even/odd nibbles
    state = np.array([[int(c, 16) for c in row[::-1]] for row in ROUNDS_IN], dtype=np.uint8)
    X_even = state[:, 0::2]   # (x_0, x_2, ..., x_30) -- the s-box inputs
    X_odd = state[:, 1::2]   # (x_1, x_3, ..., x_31)

    num_rounds = len(state) - 1
    sbox_in = X_even[:num_rounds].copy()
    sbox_out = np.zeros((num_rounds, 16), dtype=np.uint8)
    for r in range(num_rounds):
        # generalized-Feistel consistency: the s-box inputs become the next X_odd
        assert np.array_equal(X_odd[r + 1], X_even[r]), f"Feistel mismatch at round {r}"
        V = np.roll(X_odd[r], -6)                              # X_odd <<< 6 nibbles
        sbox_out[r] = nibble_perm_inverse((X_even[r + 1] ^ V).astype(np.uint8))

    # sanity check: every s-box transition is possible and the total probability matches
    log2_p = 0.0
    for r in range(num_rounds):
        for i in range(16):
            d = int(DDT[sbox_in[r, i], sbox_out[r, i]])
            assert d > 0, f"invalid s-box transition at round {r}, nibble {i}"
            log2_p += np.log2(d / 16)
    assert round(log2_p) == -122, f"expected 2^-122, got 2^{round(log2_p)}"

    dst_file = Path(__file__).with_suffix('.npz')
    print(f'Writing to {dst_file} (num_rounds={num_rounds}, probability=2^{round(log2_p)})')
    np.savez(dst_file, sbox_in=sbox_in[:, POL], sbox_out=sbox_out[:, POL])

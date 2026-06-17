import numpy as np
from sat_toolkit.formula import XorCNF, CNF, Clause
from sat_toolkit.formula import XorCNF, CNF, Truthtable


def truthtable_to_cnf(tt: Truthtable) -> CNF:

    cnf = tt.to_cnf()

    # don't return the cached version, because it's mutable
    return CNF(cnf)


def lut_to_cnf(lut: np.ndarray) -> CNF:
    truthtable = Truthtable.from_lut(lut.T.flatten())
    print(f"truthtable: {truthtable}")
    return truthtable_to_cnf(truthtable)


sbox = np.array([int(x, 16) for x in "c56b90ad3ef84712"], dtype=np.uint8)


def _get_cnf(sbox: np.ndarray, x_set: np.ndarray) -> CNF:
    lut = np.zeros((len(sbox), len(sbox)), dtype=np.uint8)
    lut[x_set, sbox[x_set]] = 1  # set indicated by characteristic
    print(f"x_set: {x_set}, sbox[x_set]: {sbox[x_set]}")
    assert lut.sum() == len(x_set)
    return lut_to_cnf(lut)


def _get_sbox_cnf(delta_in, delta_out) -> CNF:
    x = np.arange(len(sbox), dtype=np.uint8)
    # print(f"delta_in: {delta_in}, delta_out: {delta_out}")
    x_set, = np.where(
        sbox[x] ^ sbox[x ^ delta_in] == delta_out)  # values for which the differential characteristic occurs
    cnf = _get_cnf(sbox, x_set)
    print(cnf)
    return cnf


delta_in = int("b", base=16)
delta_out = 8


for i in range(16):
    out1 = sbox[i]
    out2 = sbox[i ^ delta_in]
    if out1 ^ out2 == delta_out:
        print(f"({i}, {out1}; {i ^ delta_in}, {out2})")

_get_sbox_cnf(delta_in, delta_out)
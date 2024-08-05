from autodiver.cipher_model import count_solutions
from copy import copy
from typing import Any
from shutil import which

import numpy as np
from sat_toolkit.formula import CNF, XorCNF
import pytest

from autodiver.pyjamask.util import load_state, unload_state
from autodiver.pyjamask.pyjamask96_model import Pyjamask96Characteristic, Pyjamask_Longkey, Pyjamask_with_Keyschedule
from autodiver_ciphers.pyjamask.pyjamask_cipher import pypyjamask_96_enc_longkey, pypyjamask_96_enc

approxmc = which("approxmc")


def expand_roundkeys(roundkeys: np.ndarray[Any, np.dtype[np.uint32]], numrounds)  -> np.ndarray[Any, np.dtype[np.uint8]]:
    new_roundkeys = np.zeros([numrounds + 1, 4], dtype=np.uint32)
    new_roundkeys[:, 0:3] = roundkeys[:, 0:3]
    return new_roundkeys.view(np.uint8)


def test_zero_characteristic_pyjama96_longkey():
    numrounds = 14
    sbi = sbo = np.zeros((numrounds, 3), dtype=np.uint32)

    char = Pyjamask96Characteristic(sbi, sbo)
    pyjama = Pyjamask_Longkey(char)

    model = pyjama.solve()

    round_keys = model.round_keys # type: ignore
    pt = model.pt # type: ignore
    ct = unload_state(model.ct, 3) # type: ignore

    # print(sbo[0])
    # sbo_ref = pypyjamask_96_subbytes(sbi[0])
    # print(np.array(sbo_ref).view(np.uint32))
    # print("\n")

    round_keys_model = expand_roundkeys(round_keys, numrounds)
    pt_model = unload_state(pt, 3)

    ct_ref = pypyjamask_96_enc_longkey(pt_model, round_keys_model.flatten(), numrounds)

    assert bytearray(ct) == ct_ref



def load_char(numrounds: int) -> Pyjamask96Characteristic:
    sbox_in = np.array([
        [0x00000000, 0x00000000, 0x00a04e67],
        [0x00000000, 0xa900010a, 0x00000000],
        [0x2040b886, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x04010c62],
        [0x00000000, 0x0a3a0841, 0x00000000],
    ], dtype=np.uint32)
    sbox_out = np.roll(sbox_in, -1, axis=1)

    if numrounds > len(sbox_in):
        raise ValueError("can have at most 5 rounds")

    return Pyjamask96Characteristic(sbox_in[:numrounds], sbox_out[:numrounds])


def test_nonzero_characteristic_pyjama96_longkey():
    numrounds = 5
    char = load_char(numrounds)
    pyjama = Pyjamask_Longkey(char)
    model = pyjama.solve()

    round_keys = model.round_keys  # type: ignore
    pt = model.pt  # type: ignore
    ct = unload_state(model.ct, 3) # type: ignore

    round_keys_model = expand_roundkeys(round_keys, numrounds)
    pt_model = unload_state(pt, 3)

    ct_ref = pypyjamask_96_enc_longkey(pt_model, round_keys_model.flatten(), numrounds)

    assert bytearray(ct) == ct_ref

@pytest.mark.skipif(approxmc is None, reason="approxmc not found")
def test_count_pyjama96():
    num_rounds = 2
    sbi = sbo = np.zeros((num_rounds, 3), dtype=np.uint32)
    char = Pyjamask96Characteristic(sbi, sbo)
    pyjama = Pyjamask_Longkey(char)

    cnf_new = copy(pyjama.cnf)
    cnf_new += CNF.create_xor(pyjama.key.flatten())
    cnf_new += CNF.create_xor(pyjama.pt.flatten())
    sol_count = count_solutions(cnf_new, epsilon=0.2, delta=0.8)
    assert sol_count == 1

    # cnf_all = copy(pyjama.cnf)
    # sol_count = count_solutions(cnf_all, epsilon=0.2, delta=0.8)
    # assert sol_count == 2**96 * 2**(96*3)

def test_zero_characteristic_pyjama96():
    numrounds = 14
    sbi = sbo = np.zeros((numrounds, 3), dtype=np.uint32)

    char = Pyjamask96Characteristic(sbi, sbo)
    pyjama = Pyjamask_with_Keyschedule(char)

    model = pyjama.solve(seed=912)

    key = model.key # type: ignore
    pt = model.pt # type: ignore
    ct = unload_state(model.ct, 3) # type: ignore

    # print(sbo[0])
    # sbo_ref = pypyjamask_96_subbytes(sbi[0])
    # print(np.array(sbo_ref).view(np.uint32))
    # print("\n")

    #round_keys_model = expand_roundkeys(round_keys, numrounds)
    pt_model = unload_state(pt, 3)
    key_model = unload_state(key, 4)

    ct_ref = pypyjamask_96_enc(pt_model, key_model, numrounds, numrounds)

    print(ct.view(np.uint8))
    print(np.array(ct_ref).view(np.uint8))

    assert bytearray(ct) == ct_ref


def test_nonzero_characteristic_pyjama96():
    numrounds = 3
    char = load_char(numrounds)
    pyjama = Pyjamask_with_Keyschedule(char)

    model = pyjama.solve(seed=912)

    key = model.key # type: ignore
    pt = model.pt # type: ignore
    ct = unload_state(model.ct, 3) # type: ignore

    print(pt.view(np.uint32))

    # print(sbo[0])
    # sbo_ref = pypyjamask_96_subbytes(sbi[0])
    # print(np.array(sbo_ref).view(np.uint32))
    # print("\n")

    #round_keys_model = expand_roundkeys(round_keys, numrounds)
    pt_model = unload_state(pt, 3)
    key_model = unload_state(key, 4)

    ct_ref = pypyjamask_96_enc(pt_model, key_model, numrounds, numrounds)

    print(ct.view(np.uint8))
    print(np.array(ct_ref).view(np.uint8))

    assert bytearray(ct) == ct_ref

if __name__ == "__main__":
    test_nonzero_characteristic_pyjama96_longkey()

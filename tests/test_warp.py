from random import randint
from pathlib import Path

import numpy as np
import pytest
from shutil import which
from autodiver.cipher_model import count_solutions
from autodiver.warp128.warp128_model import WARP128, WarpCharacteristic
from autodiver_ciphers.warp128.warp_cipher import warp_enc
from autodiver.warp128.util import get_round_in_out, perm_nibble_inv

from sat_toolkit.formula import CNF
from icecream import ic

REPO_ROOT = Path(__file__).resolve().parent.parent


def print_state(S, state = "s"):
    print(state, ":", end = " ")
    for s in S:
        print(hex(s)[2:], end = "")
    print("")

def Add(A, B):
    assert A.shape == B.shape
    state = []
    for i in range(len(A)):
        s = A[i] ^ B[i]
        state.append(s)
    return np.asarray(state, np.uint8)



def read_hex(s: str) -> np.ndarray:
    a = [int(x, 16) for x in s]
    return np.array(a, dtype=np.uint8)

testvectors = [
    (read_hex("0123456789ABCDEFFEDCBA9876543210"), read_hex("0123456789ABCDEFFEDCBA9876543210"), read_hex("24CE0A8EFD9F32DE529D5FDF45703A8D")),
    (read_hex("00112233445566778899AABBCCDDEEFF"), read_hex("0123456789ABCDEFFEDCBA9876543210"), read_hex("923C64F92827EE62B9667DD2548FB12C")),
    (read_hex("AF6CDD90FC5A6EAA897BCD1208D391E1"), read_hex("0ACD022F680A547FEE03C0867B09E3D7"), read_hex("6123995F1924D31425641ACDD058DD46")),
]

@pytest.mark.parametrize("pt,key,ct_ref", testvectors)
def test_tv(pt, key, ct_ref):
    print_state(pt, "M")
    print_state(key, "K")
    print_state(ct_ref, "C")

    ct = warp_enc(pt, key, 41)
    print_state(ct, "C")
    assert np.all(ct == ct_ref)

def test_zero_characteristic():
    numrounds = 3
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = WarpCharacteristic(sbi, sbo)
    char.sbox_in = sbi
    char.sbox_out = sbo
    char.num_rounds = numrounds
    warp = WARP128(char)
    # print(warp.cnf)


    num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 128)

    for bit_var in warp.key.flatten():
        warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])

    num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << 128

    for bit_var in warp.sbox_in[0].flatten():
        warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    for bit_var in warp.X.flatten():
        warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])

    model = warp.solve(seed=9399)

    key = model.key # type: ignore
    print_state(key)
    pt = model.pt # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(warp.sbox[sbi[:warp.num_rounds]] == sbo)

    rout = get_round_in_out(numrounds, model)

    # print_state(rin[0])
    # assert np.all(rin[0] == pt)
    print_state(pt)
    for r, round_out in enumerate(rout):
        ref = warp_enc(pt, key, r+1)
        print("----------------------------------")
        # print_state(sbi[r])
        # if r != warp.num_rounds - 1:
        #     print_state(perm_nibble_16_inv(sbi[r+1]))
        # else:
        #     print_state(model.Y)
        print_state(ref)
        print_state(round_out)
        print("----------------------------------")
        assert np.all(round_out == ref)

    num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1

def test_nonzero_characteristic():
    char = WarpCharacteristic.load(REPO_ROOT / "trails" / "warp" / "warp_KY22_r18_table_8.npz")
    char.truncate_rounds((0, 8))

    warp = WARP128(char)
    model = warp.solve(seed=3983)

    key = model.key # type: ignore
    print_state(key)
    pt = model.pt # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    rout = get_round_in_out(warp.num_rounds, model)

    assert np.all(warp.sbox[sbi[:warp.num_rounds]] == sbo)

    print_state(pt, "pt")
    delta0 = char.round_in[0]
    for r in range(1, char.num_rounds):
        print(f' round {r} '.center(80, '='))
        ref = warp_enc(pt, key, r)
        ref_xor = warp_enc(Add(pt, delta0), key, r)

        assert np.all(rout[r-1] == ref)

        found_diff = Add(ref, ref_xor)
        print_state(found_diff)
        expected_diff = perm_nibble_inv(char.round_in[r])
        print_state(expected_diff)
        assert np.all(expected_diff == found_diff)

if __name__ == "__main__":
    test_zero_characteristic()
    test_nonzero_characteristic()

    # for tv in midori64_testvectors:
    #     test_tv(*tv)

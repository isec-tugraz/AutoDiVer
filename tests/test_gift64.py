from random import randint

import numpy as np
import pytest
from shutil import which

from autodiver.cipher_model import count_solutions
from autodiver.gift.gift_model import Gift64, Gift64Characteristic
from autodiver_ciphers.gift.gift64_cipher import gift64_enc

from sat_toolkit.formula import CNF

def print_state(S, state = "s"):
    print(state, ":", end = " ")
    for s in S:
        print(hex(s)[2:], end = "")
    print("")

def read_hex(s: str) -> np.ndarray:
    a = [int(x, 16) for x in s]
    a.reverse()  #test vectors are given in MSB first
    return np.array(a, dtype=np.uint8)

testvectors = [
    (read_hex("c450c7727a9b8a7d"), read_hex("bd91731eb6bc2713a1f9f6ffc75044e7"), read_hex("e3272885fa94ba8b")),
    (read_hex("fedcba9876543210"), read_hex("fedcba9876543210fedcba9876543210"), read_hex("c1b71f66160ff587"))
]

@pytest.mark.parametrize("pt,key,ct_ref", testvectors)
def test_tv(pt, key, ct_ref):
    print_state(pt, "M")
    print_state(key, "K")
    print_state(ct_ref, "C")

    ct = gift64_enc(pt, key, 28)
    print_state(ct, "C")
    assert np.all(ct == ct_ref)


def test_zero_characteristic():
    numrounds = 5
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = Gift64Characteristic(sbi, sbo)
    gift = Gift64(char)


    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 64)

    for bit_var in gift.key.flatten():
        gift.cnf += CNF([bit_var * (-1)**randint(0,1), 0])

    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << 64

    for bit_var in gift.sbox_in[0].flatten():
        gift.cnf += CNF([bit_var * (-1)**randint(0,1), 0])

    model = gift.solve(seed=1726)

    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore

    assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)

    for r, round_sbi in enumerate(sbi):
        ref = gift64_enc(sbi[0], key, r)
        print(f'{ref}')
        print(f'{round_sbi}')
        assert np.all(round_sbi == ref)

    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1

def test_nonzero_characteristic():
    char = (
        ("0000000c00000006", "0000000200000002"),
        ("0000000002020000", "0000000005050000"),
        ("0000005000000050", "0000002000000020"),
        ("0000000000000202", "0000000000000505"),
        ("0000000500000005", "0000000200000002"),
        ("0000000002020000", "0000000005050000"),
        ("0000005000000050", "0000002000000020"),
        ("0000000000000202", "0000000000000505"),
        ("0000000500000005", "0000000f0000000f"),
    )
    sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
    sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)

    char = Gift64Characteristic(sbi_delta, sbo_delta)

    gift = Gift64(char)
    model = gift.solve(seed=6804)

    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore

    for r, round_sbi in enumerate(sbi):
        ref = gift64_enc(sbi[0], key, r)
        ref_xor = gift64_enc(sbi[0] ^ sbi_delta[0], key, r)
        assert np.all(round_sbi == ref)
        if r < gift.num_rounds - 1:
            assert np.all(round_sbi ^ sbi_delta[r] == ref_xor)

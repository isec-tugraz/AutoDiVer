from random import randint
import pytest
import numpy as np
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.gift128.gift_model import Gift128
from differential_verification.gift128.gift_cipher import gift128_enc
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
    (read_hex("00000000000000000000000000000000"), read_hex("00000000000000000000000000000000"), read_hex("cd0bd738388ad3f668b15a36ceb6ff92")),
    (read_hex("fedcba9876543210fedcba9876543210"), read_hex("fedcba9876543210fedcba9876543210"), read_hex("8422241a6dbf5a9346af468409ee0152")),
    (read_hex("e39c141fa57dba43f08a85b6a91f86c1"), read_hex("d0f5c59a7700d3e799028fa9f90ad837"), read_hex("13ede67cbdcc3dbf400a62d6977265ea")),
]
@pytest.mark.parametrize("pt,key,ct_ref", testvectors)
def test_tv(pt, key, ct_ref):
    print_state(pt, "M")
    print_state(key, "K")
    print_state(ct_ref, "C")
    ct = gift128_enc(pt, key, 40)
    print_state(ct, "C")
    assert np.all(ct == ct_ref)
def test_zero_characteristic():
    numrounds = 2
    sbi = sbo = np.zeros((numrounds, 32), dtype=np.uint8)
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
    char.sbox_in = sbi
    char.sbox_out = sbo
    char.num_rounds = numrounds
    gift = Gift128(char)
    # print(gift.cnf)
    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 128)
    for bit_var in gift.key.flatten():
        gift.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << 128
    for bit_var in gift.sbox_in[0].flatten():
        gift.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    model = gift.solve()
    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    print_state(key, "key")
    assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)
    print_state(sbi[0])
    for r, round_sbi in enumerate(sbi):
        ref = gift128_enc(sbi[0], key, r)
        print_state(ref)
        print_state(round_sbi)
        assert np.all(round_sbi == ref)
    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1
def test_nonzero_characteristic():
    char = (
        ("00000000000000000000060700000000", "00000000000000000000020800000000"),
        ("00000a00000000000000000000000000", "00000100000000000000000000000000"),
        ("00000000000000000000000001000000", "00000000000000000000000008000000"),
        ("00000000000000000000008000000000", "00000000000000000000003000000000"),
        ("00000000000000000000010000000200", "00000000000000000000060000000600"),
        ("00000202000004040000000000000000", "00000505000005050000000000000000"),
        ("00000000050500000000000005050000", "00000000020800000000000002080000"),
        # ("00a000a0000000000000000000000000", "00100010000000000000000000000000"),
        # ("00000000000000001100000000000000", "00000000000000008800000000000000"),
        # ("00000000000000000000800000008000", "00000000000000000000300000003000"),
        # ("00000101000002020000000000000000", "00000505000005050000000000000000"),
        # ("00000000050500000000000005050000", "00000000080200000000000008020000"),
        # ("000000000000000000a000a000000000", "00000000000000000010001000000000"),
        # ("00000000000000000000110000000000", "00000000000000000000c90000000000"),
        # ("000000000000000000000c0000000900", "00000000000000000000080000000100"),
        # ("00000000000000000000080000000001", "00000000000000000000030000000008"),
        # ("00000208000000000000000000000100", "00000603000000000000000000000800"), # <---
        # ("02000000050000000200000800000000", "060000000f0000000600000300000000"), # <--- these two rounds are UNSAT
    )
    np.set_printoptions(formatter={'int': lambda x: f'{x:01x}'})
    sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
    sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
    print(f'ddt probability: 2^{char.log2_ddt_probability(Gift128.ddt):.1f}')
    gift = Gift128(char)
    model = gift.solve()
    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)
    assert np.all(gift.sbox[sbi[:gift.num_rounds] ^ char.sbox_in] == sbo ^ char.sbox_out)
    for r, round_sbi in enumerate(sbi):
        ref = gift128_enc(sbi[0], key, r)
        ref_xor = gift128_enc(sbi[0] ^ sbi_delta[0], key, r)
        assert np.all(round_sbi == ref)
        if r < gift.num_rounds - 1:
            print(f"round {r:2d}: {ref_xor ^ round_sbi ^ sbi_delta[r]}")
            assert np.all(round_sbi ^ sbi_delta[r] == ref_xor)
from random import randint
import numpy as np
import pytest
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.warp128.warp128_model import WARP128
from differential_verification.warp128.warp_cipher import warp_enc
from differential_verification.warp128.util import get_round_in_out
from sat_toolkit.formula import CNF
from icecream import ic
def print_state(S, state = "s"):
    print(state, ":", end = " ")
    for s in S:
        print(hex(s)[2:], end = "")
    print("")
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
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
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
    model = warp.solve()
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
# def test_nonzero_characteristic():
#     # characteristic from https://doi.org/10.1109/ACCESS.2020.2995795S (Figure 3)
#     # with alpha = beta = 1
#     char = (
#         ("1000000000100000", "2000000000200000"),
#         ("2200000000000000", "4100000000000000"),
#         ("0444111000000000", "0222222000000000"),
#         ("2202020202022202", "4101040101011104"),
#         ("0400001100011100", "0200002200022200"),
#     )
#     sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
#     sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
#     sbi_delta = sbi_delta.reshape(-1, 4, 4).swapaxes(-1, -2)
#     sbo_delta = sbo_delta.reshape(-1, 4, 4).swapaxes(-1, -2)
#     ic(sbi_delta[1])
#     ic(sbo_delta[1])
#     char = DifferentialCharacteristic(sbi_delta, sbo_delta)
#     midori = Midori64(char)
#     model = midori.solve()
#     key = model.key # type: ignore
#     # print(f'{key = }')
#     key0 = matrix_as_uint64(key[0])
#     key1 = matrix_as_uint64(key[1])
#     # print(f'{hex(key0) = }', f'{hex(key1) = }')
#     sbi = model.sbox_in # type: ignore
#     sbo = model.sbox_out # type: ignore
#     # we need to add the key here in post-processing
#     pt = matrix_as_uint64(sbi[0]) ^ key0 ^ key1
#     sbi_delta0 = matrix_as_uint64(sbi_delta[0])
#     for r in range(1, char.num_rounds):
#         print(f' round {r} '.center(80, '='))
#         ref = midori64_enc(pt, key0, key1, r)
#         ref_xor = midori64_enc(pt ^ sbi_delta0, key0, key1, r)
#         # we need to add the key here in post-processing
#         out = matrix_as_uint64(sbo[r - 1]) ^ key0 ^ key1
#         assert out == ref
#         print(f"{ref ^ ref_xor = :016x}")
#         expected_diff = matrix_as_uint64(sbo_delta[r - 1])
#         print(f"{expected_diff = :016x}")
#         print(f"{midori64_mc(midori64_sr(ref ^ ref_xor))   = :016x}")
#         print(f"{midori64_mc(midori64_sr(expected_diff))   = :016x}")
#         ic(out, ref_xor)
#         assert expected_diff == ref ^ ref_xor
if __name__ == "__main__":
    test_zero_characteristic()
    # test_nonzero_characteristic()
    # for tv in midori64_testvectors:
    #     test_tv(*tv)
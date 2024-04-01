from random import randint
import numpy as np
import pytest
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.warp128.warp128_model import WARP128
from differential_verification.warp128.warp_cipher import warp_enc
from sat_toolkit.formula import CNF
from icecream import ic
# midori64_testvectors = [
#     (0x0000000000000000, 0x00000000000000000000000000000000, 0x3c9cceda2bbd449a),
#     (0x42c20fd3b586879e, 0x687ded3b3c85b3f35b1009863e2a8cbf, 0x66bcdc6270d901cd),
# ]
# @pytest.mark.parametrize("pt,key,ct_ref", midori64_testvectors)
# def test_tv(pt, key, ct_ref):
#     print(f'{pt = :016x}', f'{key = :032x}', f'{ct_ref = :016x}')
#     key0 = key >> 64
#     key1 = key & 0xFFFFFFFFFFFFFFFF
#     ct = midori64_enc(pt, key0, key1, 16)
#     print(' ' * 65 + f'{ct = :016x}')
#     print(f'{ct = :016x}')
#     print(f'{ct ^ ct_ref = :016x}')
#     assert ct == ct_ref
def test_zero_characteristic():
    numrounds = 2
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
    char.sbox_in = sbi
    char.sbox_out = sbo
    char.num_rounds = numrounds
    warp = WARP128(char)
    # print(warp.cnf)
    num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 128)
    # for bit_var in warp.key.flatten():
    #     warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    # num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1 << 128
    # for bit_var in warp.sbox_in[0].flatten():
    #     warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    # model = warp.solve()
    # key = model.key # type: ignore
    # sbi = model.sbox_in # type: ignore
    # sbo = model.sbox_out # type: ignore
    # assert np.all(warp.sbox[sbi[:warp.num_rounds]] == sbo)
    # for r, round_sbi in enumerate(sbi):
    #     ref = warp128_enc(sbi[0], key, r)
    #     assert np.all(round_sbi == ref)
    # num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1
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
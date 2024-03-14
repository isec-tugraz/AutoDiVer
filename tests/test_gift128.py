from random import randint
import numpy as np
from src.differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from src.differential_verification.gift128.gift_model import Gift128
from src.differential_verification.gift128.gift_cipher import gift128_enc
from sat_toolkit.formula import CNF
def test_zero_characteristic():
    numrounds = 1
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
    assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)
    for r, round_sbi in enumerate(sbi):
        ref = gift128_enc(sbi[0], key, r)
        assert np.all(round_sbi == ref)
    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1
# def test_nonzero_characteristic():
#     char = (
#         ("0000000c00000006", "0000000200000002"),
#         ("0000000002020000", "0000000005050000"),
#         ("0000005000000050", "0000002000000020"),
#         ("0000000000000202", "0000000000000505"),
#         ("0000000500000005", "0000000200000002"),
#         ("0000000002020000", "0000000005050000"),
#         ("0000005000000050", "0000002000000020"),
#         ("0000000000000202", "0000000000000505"),
#         ("0000000500000005", "0000000f0000000f"),
#     )
#     sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
#     sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
#     char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
#     char.sbox_in = sbi_delta
#     char.sbox_out = sbo_delta
#     char.num_rounds = len(sbi_delta)
#     gift = Gift64(char)
#     model = gift.solve()
#     key = model.key # type: ignore
#     sbi = model.sbox_in # type: ignore
#     sbo = model.sbox_out # type: ignore
#     for r, round_sbi in enumerate(sbi):
#         ref = gift64_enc(sbi[0], key, r)
#         ref_xor = gift64_enc(sbi[0] ^ sbi_delta[0], key, r)
#         assert np.all(round_sbi == ref)
#         if r < gift.num_rounds - 1:
#             assert np.all(round_sbi ^ sbi_delta[r] == ref_xor)
#     print('sanity check 2 passed')
if __name__ == "__main__":
    test_zero_characteristic()
    # test_nonzero_characteristic()
from random import randint
import numpy as np
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.midori128.midori128_model import Midori128
# from differential_verification.midori128.midori_cipher import midori128_enc
from sat_toolkit.formula import CNF
#0th bit is the LSB
def print_state(key):
    for k in key:
        print(hex(k)[2:], end = " ")
    print("")
def nibble_to_byte(key_arr):
    key = []
    key_arr_str = [str(hex(x)[2:]) for x in key_arr]
    # print(key_arr_str)
    for i in range(16):
        kk = key_arr_str[2*i] + key_arr_str[2*i + 1]
        kk = int(kk, 16)
        key.append(kk)
    key = np.asarray(key, dtype=np.uint8)
    return key
def test_zero_characteristic():
    numrounds = 2
    sbi = sbo = np.zeros((numrounds, 32), dtype=np.uint8)
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
    char.sbox_in = sbi
    char.sbox_out = sbo
    char.num_rounds = numrounds
    midori = Midori128(char)
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 128)
    for bit_var in midori.key.flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    # num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1 << 128
    for bit_var in midori.sbox_in[0].flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    model = midori.solve()
    key = model.key # type: ignore
    print(f'{key = }')
    key = nibble_to_byte(key[0])
    print_state(key)
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    # assert np.all(midori.sbox[sbi[:midori.num_rounds]] == sbo)
    # print(sbi[0])
    sbi0 = nibble_to_byte(sbi[0])
    print_state(sbi0)
    for r, round_sbi in enumerate(sbi):
        sbiR = nibble_to_byte(sbi[r])
        ref = midori128_enc(sbi0, key, r)
        print_state(sbiR)
        print_state(ref)
        assert np.all(sbiR == ref)
        print("--------------------------------------")
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1
# def test_nonzero_characteristic():
#     char = (
#         ("1000000000100000", "2000000000200000"),
#         ("2200000000000000", "4100000000000000"),
#         ("0444111000000000", "0222222000000000"),
#         ("2202020202022202", "4101040101011104"),
#         ("0400001100011100", "0200002200022200"),
#     )
#     sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
#     sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
#     char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
#     char.sbox_in = sbi_delta
#     char.sbox_out = sbo_delta
#     char.num_rounds = len(sbi_delta)
#     midori = Midori64(char)
#     model = midori.solve()
#     key = model.key # type: ignore
#     print(f'{key = }')
#     key0 = nibble_to_block(key[0])
#     key1 = nibble_to_block(key[1])
#     print(f'{hex(key0) = }', f'{hex(key1) = }')
#     sbi = model.sbox_in # type: ignore
#     sbo = model.sbox_out # type: ignore
#     sbi0 = nibble_to_block(sbi[0])
#     sbi_delta0 = nibble_to_block(sbi_delta[0])
#     for r, round_sbi in enumerate(sbi):
#         ref = midori64_enc(sbi0, key0, key1, r)
#         ref_xor = midori64_enc(sbi0 ^ sbi_delta0, key0, key1, r)
#         sbiR = nibble_to_block(sbi[r])
#         assert sbiR == ref
#         if r < midori.num_rounds - 1:
#             sbi_deltaR = nibble_to_block(sbi_delta[r])
#             outD = sbiR ^ sbi_deltaR
#             print(f'{hex(ref) = }', f'{hex(sbiR) = }')
#             print(f'{hex(ref_xor) = }', f'{hex(outD) = }')
#             assert outD == ref_xor
#     print('sanity check 2 passed')
if __name__ == "__main__":
    test_zero_characteristic()
    # test_nonzero_characteristic()
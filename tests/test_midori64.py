from random import randint
import numpy as np
from cipher_model import DifferentialCharacteristic, count_solutions
from midori.midori import Midori64
from sat_toolkit.formula import CNF
from pyximport import install
install()
from midori.midori_cipher import midori64_enc
def nibble_to_block(key_arr):
    key = 0x00000000000000000
    key_arr_str = [str(hex(x)[2:]) for x in key_arr]
    key = ''.join(key_arr_str)
    key = int(key, 16)
    # for i in range(16):
    #     key = (key << 4) | key_arr[15 - i]
    return key
def test_zero_characteristic():
    numrounds = 3
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
    char.sbox_in = sbi
    char.sbox_out = sbo
    char.num_rounds = numrounds
    midori = Midori64(char)
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 64)
    for bit_var in midori.key.flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << 64
    # for bit_var in midori.sbox_in[0].flatten():
    #     midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    # model = midori.solve()
    # key = model.key # type: ignore
    # print(f'{key = }')
    # key0 = nibble_to_block(key[0])
    # key1 = nibble_to_block(key[1])
    # print(f'{hex(key0) = }', f'{hex(key1) = }')
    # sbi = model.sbox_in # type: ignore
    # sbo = model.sbox_out # type: ignore
    # assert np.all(midori.sbox[sbi[:midori.num_rounds]] == sbo)
    # for r, round_sbi in enumerate(sbi):
    #     sbi0 = nibble_to_block(sbi[0])
    #     ref = midori64_enc(sbi0, key0, key1, r)
    #     assert np.all(round_sbi == ref)
    # num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1
if __name__ == "__main__":
    test_zero_characteristic()
from random import randint
import numpy as np
from cipher_model import DifferentialCharacteristic, count_solutions
from midori64.midori64_model import Midori64
from sat_toolkit.formula import CNF
from pyximport import install
install()
from midori64.midori_cipher import midori64_enc
#0th bit is the LSB
def nibble_to_block(key_arr):
    key_arr_str = [str(hex(x)[2:]) for x in key_arr]
    key = ''.join(key_arr_str)
    key = int(key, 16)
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
    for bit_var in midori.sbox_in[0].flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    model = midori.solve()
    key = model.key # type: ignore
    print(f'{key = }')
    key0 = nibble_to_block(key[0])
    key1 = nibble_to_block(key[1])
    print(f'{hex(key0) = }', f'{hex(key1) = }')
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(midori.sbox[sbi[:midori.num_rounds]] == sbo)
    sbi0 = nibble_to_block(sbi[0])
    for r, round_sbi in enumerate(sbi):
        sbiR = nibble_to_block(sbi[r])
        ref = midori64_enc(sbi0, key0, key1, r)
        print(f'{sbi0 = }', f'{ref = }', f'{sbiR = }')
        assert sbiR == ref
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1
def test_nonzero_characteristic():
    char = (
        ("1000000000100000", "2000000000200000"),
        ("2200000000000000", "4100000000000000"),
        ("0444111000000000", "0222222000000000"),
        ("2202020202022202", "4101040101011104"),
        ("0400001100011100", "0200002200022200"),
    )
    sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
    sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
    char.sbox_in = sbi_delta
    char.sbox_out = sbo_delta
    char.num_rounds = len(sbi_delta)
    midori = Midori64(char)
    model = midori.solve()
    key = model.key # type: ignore
    print(f'{key = }')
    key0 = nibble_to_block(key[0])
    key1 = nibble_to_block(key[1])
    print(f'{hex(key0) = }', f'{hex(key1) = }')
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    sbi0 = nibble_to_block(sbi[0])
    sbi_delta0 = nibble_to_block(sbi_delta[0])
    for r, round_sbi in enumerate(sbi):
        ref = midori64_enc(sbi0, key0, key1, r)
        ref_xor = midori64_enc(sbi0 ^ sbi_delta0, key0, key1, r)
        sbiR = nibble_to_block(sbi[r])
        assert sbiR == ref
        if r < midori.num_rounds - 1:
            sbi_deltaR = nibble_to_block(sbi_delta[r])
            outD = sbiR ^ sbi_deltaR
            print(f'{hex(ref) = }', f'{hex(sbiR) = }')
            print(f'{hex(ref_xor) = }', f'{hex(outD) = }')
            assert outD == ref_xor
    print('sanity check 2 passed')
if __name__ == "__main__":
    test_zero_characteristic()
    test_nonzero_characteristic()
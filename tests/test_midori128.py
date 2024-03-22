from random import randint
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.midori128.midori128_model import Midori128
from differential_verification.midori128.midori_cipher import midori128_enc
import numpy as np
import pytest
from sat_toolkit.formula import CNF
#0th bit is the LSB
def print_state(key):
    for k in key:
        print(hex(k)[2:], end = " ")
    print("")
midori128_testvectors = [
    ("00000000000000000000000000000000", "00000000000000000000000000000000", "c055cbb95996d14902b60574d5e728d6"),
    ("51084ce6e73a5ca2ec87d7babc297543", "687ded3b3c85b3f35b1009863e2a8cbf", "1e0ac4fddff71b4c1801b73ee4afc83d"),
]
midori128_testvectors = [
    (bytes.fromhex(pt), bytes.fromhex(key), bytes.fromhex(ct_ref)) for pt, key, ct_ref in midori128_testvectors
]
@pytest.mark.parametrize("pt,key,ct_ref", midori128_testvectors)
def test_tv(pt: bytes, key: bytes, ct_ref: bytes):
    # print(f'{pt = :016x}', f'{key = :032x}', f'{ct_ref = :016x}')
    # key0 = key >> 64
    # key1 = key & 0xFFFFFFFFFFFFFFFF
    ct = midori128_enc(pt, key, 20)
    # print(' ' * 65 + f'{ct = :016x}')
    # print(f'{ct = :016x}')
    # print(f'{ct ^ ct_ref = :016x}')
    assert ct == ct_ref
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
    sbi_delta = sbo_delta = np.zeros((numrounds, 32), dtype=np.uint8)
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
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
def test_nonzero_characteristic():
    char =  (
        ('00000000000200000000400000000000', '00000000000100000000200000000000'),
        ('00012000000000000000000000000000', '00021000000000000000000000000000'),
        ('00000000100240000000000040800020', '00000000800110000000000040800040'),
        ('00002040800020400000102040001020', '00001020800010200000204040002040'),
        ('00010040000000404000000040001000', '00010020000000204000000040002000'),
        ('00000000800000400080100040000020', '00000000800000200080200040000040'),
        ('00010040800020404000100000801020', '00010020800010204000200000802040'),
        ('00010040000020000080002000001000', '00010020000010000080004000002000'),
        ('00000000000100000000100000000000', '00000000000100000000200000000000'),
        ('00012000000000000000000000000000', '00021000000000000000000000000000')
    )
    char1 = (
        ('00000000001000000000100000000000', '00000000000800000000080000000000'),
        ('00080800000000000000000000000000', '00104000000000000000000000000000'),
        ('00000000101010000000000040400040', '00000000080840000000000040400008'),
        ('00000808080008080000404040004040', '00004040080040400000080840000808'),
        ('00080008000000084000000040004000', '00080040000000404000000040000800'),
        ('00000000080000080040400040000040', '00000000080000400040080040000008'),
        ('00080008080008084000400000404040', '00080040080040404000080000400808'),
        ('00080008000008000040004000004000', '00080040000040000040000800000800'),
        ('00000000000800000000400000000000', '00000000000800000000080000000000'),
        ('00080800000000000000000000000000', '00104000000000000000000000000000')
    )
    sbi_delta = np.array([[int(x, 32) for x in inp] for inp, _ in char], dtype=np.uint8)
    sbo_delta = np.array([[int(x, 32) for x in out] for _, out in char], dtype=np.uint8)
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
    midori = Midori128(char)
    model = midori.solve()
    key = model.key # type: ignore
    print(f'{key = }')
    key = nibble_to_byte(key[0])
    print_state(key)
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    sbi0 = nibble_to_byte(sbi[0])
    sbi_delta0 = nibble_to_byte(sbi_delta[0])
    X1 = sbi0 ^ sbi_delta0
    print_state(X1)
    for r, round_sbi in enumerate(sbi):
        ref = midori128_enc(sbi0, key, r)
        sbiR = nibble_to_byte(sbi[r])
        assert np.all(sbiR == ref)
        ref_xor = midori128_enc(X1, key, r)
        # if r < midori.num_rounds - 1:
        #     sbi_deltaR = nibble_to_byte(sbi_delta[r])
        #     outD = sbiR ^ sbi_deltaR
        #     print("--------------------------------------")
        #     print_state(ref)
        #     print_state(sbiR)
        #     print_state(ref_xor)
        #     print_state(outD)
        #     print("--------------------------------------")
        #     assert np.all(outD == ref_xor)
    print('sanity check 2 passed')
if __name__ == "__main__":
    test_zero_characteristic()
    test_nonzero_characteristic()
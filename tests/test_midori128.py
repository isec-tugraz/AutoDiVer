from random import seed, randint
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.midori128.midori128_model import Midori128, Midori128Characteristic
from differential_verification.midori128.midori_cipher import midori128_enc
from differential_verification.midori128.util import sr_mapping, postPermute
import numpy as np
import pytest
from sat_toolkit.formula import CNF
from icecream import ic
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
    ct = midori128_enc(pt, key, 20)
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
    seed("test_midori128::test_zero_characteristic")
    numrounds = 3
    sbi_delta = sbo_delta = np.zeros((numrounds, 4, 4), dtype=np.uint8)
    char = Midori128Characteristic(sbi_delta, sbo_delta, file_path=None)
    midori = Midori128(char)
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 128)
    for bit_var in midori.key.flatten():
        # midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
        midori.cnf += CNF([-bit_var, 0])
    # num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1 << 128
    ic(midori.sbox_in[0].shape)
    for bit_var in midori.sbox_out[0, 1:].flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
        # midori.cnf += CNF([-bit_var, 0])
    for bit_var in midori.sbox_out[0, :1].flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
        # midori.cnf += CNF([bit_var, 0])
    model = midori.solve(seed=6405)
    key = model.key # type: ignore
    key = nibble_to_byte(key[0])
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    # assert np.all(midori.sbox[sbi[:midori.num_rounds]] == sbo)
    np.set_printoptions(formatter={'int': lambda x: f'{x:02x}'})
    pt = nibble_to_byte(sbi[0]) ^ key
    # from IPython import embed; embed()
    for r in range(1, numrounds):
        sbiR = nibble_to_byte(sbo[r - 1])
        # we need to add the key here in post-processing
        out = sbiR ^ key
        ref = midori128_enc(pt, key, r)
        ref = np.array(bytearray(ref))
        ic(sbiR.reshape(4, 4), out.reshape(4, 4), ref.reshape(4, 4))
        ic(out ^ ref)
        assert np.all(out == ref)
        print("--------------------------------------")
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1
def test_nonzero_characteristic():
    seed("test_midori128::test_nonzero_characteristic")
    sbi_delta = np.array(bytearray.fromhex(
        '00002000000000800000000000410000'
        '00000000000000000000000100000000'
        '00000000008000000080000000800000'
        '01000400010004040000040401000004'
        # '00802000000024048000040000800404'
        # '00000000808004000080040580000401'
        # '01000000008000000000040000000000'
        # '00000000000000000000000080000000'
        # '00000400000000000000040000000400'
        # '80020020000200208000002080020000'
    )).reshape(-1, 4, 4)
    sbo_delta = np.array(bytearray.fromhex(
        '00000100000000010000000000010000'
        '00000000000000000000008000000000'
        '00000000000100000004000000040000'
        '80000400800024040000808020000080'
        # '00800500000004040100800000048080'
        # '00000000800104000004018004008001'
        # '80000000008000000000800000000000'
        # '00000000000000000000000004000000'
        # '00002000000000000000800000000200'
        # '84100001000100100100002804040000'
    )).reshape(-1, 4, 4)
    char = Midori128Characteristic(sbi_delta, sbo_delta, file_path=None)
    sbi_delta = char.sbox_in
    sbo_delta = char.sbox_out
    midori = Midori128(char)
    model = midori.solve(seed=8406)
    key = model.key # type: ignore
    key = nibble_to_byte(key[0])
    print_state(key)
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    sbi0 = nibble_to_byte(sbi[0])
    pt = sbi0 ^ key
    print(f'{pt = }')
    for r in range(1, len(sbi)):
        print(f'round: {r}')
        ref = midori128_enc(pt, key, r)
        ref = np.array(bytearray(ref)) ^ key
        sbiR = nibble_to_byte(sbo[r - 1])
        assert np.all(sbiR == ref)
        inD = nibble_to_byte(sbi_delta[0])
        ref_xor = midori128_enc(pt ^ postPermute(inD), key, r)
        ref_xor = np.array(bytearray(ref_xor)) ^ key
        print_state(ref)
        print_state(ref_xor)
        found_diff = ref ^ ref_xor
        print_state(found_diff)
        expected_diff = postPermute(nibble_to_byte(sbo_delta[r-1]))
        print_state(expected_diff)
        assert np.all(expected_diff == found_diff)
if __name__ == "__main__":
    test_zero_characteristic()
    test_nonzero_characteristic()
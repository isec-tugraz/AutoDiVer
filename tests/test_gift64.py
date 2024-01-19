from random import randint
import numpy as np
from cipher_model import DifferentialCharacteristic
from gift64.gift64 import Gift64
from pyximport import install
install()
from gift64.gift_cipher import gift64_enc
def test_zero_characteristic():
    numrounds = 26
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = DifferentialCharacteristic(sbi, sbo)
    gift = Gift64(char)
    for bit_var in gift.key.flatten():
        gift.cnf.append([bit_var * (-1)**randint(0,1)])
    for bit_var in gift.sbox_in[0].flatten():
        gift.cnf.append([bit_var * (-1)**randint(0,1)])
    model = gift.solve()
    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)
    for r, round_sbi in enumerate(sbi):
        ref = gift64_enc(sbi[0], key, r)
        assert np.all(round_sbi == ref)
    mantissa, exponent = gift._count_solutions(0.2, 0.8, verbosity=0)
    assert mantissa * 2**exponent == 1
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
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
    gift = Gift64(char)
    model = gift.solve()
    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    for r, round_sbi in enumerate(sbi):
        ref = gift64_enc(sbi[0], key, r)
        ref_xor = gift64_enc(sbi[0] ^ sbi_delta[0], key, r)
        assert np.all(round_sbi == ref)
        if r < gift.num_rounds - 1:
            assert np.all(round_sbi ^ sbi_delta[r] == ref_xor)
    print('sanity check 2 passed')
from copy import copy
import random

import pytest
import numpy as np
from typing import Any
from sat_toolkit.formula import CNF, XorCNF

from autodiver.skinny.skinny_model import Skinny64, Skinny64Characteristic, Skinny64LongKey
from autodiver.skinny.constants import do_mix_cols, expanded_rc, do_inv_mix_cols, do_shift_rows, expanded_rc, update_tweakey, tweakey_mask
from autodiver_ciphers.skinny.skinny import skinny64_enc_ecb


def matrix_to_packed_uint8(matrix: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[[Any, np.dtype[np.uint32]]]:
    assert np.all(matrix >= 0) and np.all(matrix < 16)
    assert matrix.shape == (4, 4)
    flat = matrix.ravel()

    packed_mat = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        packed_mat[i] = flat[i*2] << 4 | flat[i*2 + 1]
    return packed_mat

def pad(s: bytearray) -> bytearray:
    nibbles = []
    for byte in s:
        nibbles.append(byte >> 4)
        nibbles.append(byte & 0xf)
    return bytearray(nibbles)

def test_zero_characteristic():
    numrounds = 4
    sbox_in_delta = sbox_out_delta = np.zeros((numrounds, 4, 4), dtype=np.uint8) # important here for the real characteristics: we only use the first 4 bits of the uint8_t (4-bit nibbles)
    tweakeys = np.zeros((numrounds, 3, 4, 4), dtype=np.uint8)

    char = Skinny64Characteristic(sbox_in_delta, sbox_out_delta, tweakeys)
    cipher = Skinny64(char)

    model = cipher.solve(seed=9789)
    sbox_in = model.sbox_in #type: ignore
    sbox_out = model.sbox_out #type: ignore

    key, tk2, tk3 = model.key, model.tk2, model.tk3 #type: ignore
    pt = matrix_to_packed_uint8(sbox_in[0])
    ct = matrix_to_packed_uint8(sbox_in[-1])
    tweakey = np.array([matrix_to_packed_uint8(key), matrix_to_packed_uint8(tk2), matrix_to_packed_uint8(tk3)]).flatten()

    sbox_in = sbox_in[:-1]
    sbox = cipher.sbox

    assert np.all(sbox[sbox_in] == sbox_out)

    # sanity check result
    rtk = np.array([key, tk2, tk3])
    for i in range(numrounds):
        sbi = sbox_in[i]
        sbo = sbox_out[i]
        assert np.all(sbo == sbox[sbi])

    ct_ref = np.array(bytearray(skinny64_enc_ecb(pt, tweakey, numrounds)))

    print(f"pt: {pt}")
    print(f"ct: {ct}")
    print(f"ct_ref: {ct_ref}")
    print(f"diff: {ct_ref ^ ct}")
    assert np.all(ct_ref == ct)


@pytest.mark.parametrize("rounds", [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 10)])
def test_nonzero_characteristic(rounds):
    numrounds = len(range(rounds.start, rounds.stop))
    sbox_in = np.array(pad(bytearray.fromhex(
        "0000000140000004"
        "0000000000000020"
        "010D000D0000000D"
        "0020000020000000"
        "0000003000300000"
        "0000C000000C0000"
        "0200000000000200"
        "3000000000000000"
        "0000000000000000"
        "0000000000000000"
        "0010001000000010"
        "0A00000000050000"
        "00000A0000000000"
        "0000000000000000"
        "0000000000040000"
    ))).reshape(15, 4, 4)[rounds]

    sbox_out = np.array(pad(bytearray.fromhex(
        "0000000820000002"
        "0000000000000010"
        "0A0E000200000002"
        "0030000030000000"
        "000000C000C00000"
        "0000200000020000"
        "0500000000000300"
        "D000000000000000"
        "0000000000000000"
        "0000000000000000"
        "00800090000000A0"
        "0A000000000A0000"
        "00000A0000000000"
        "0000000000000000"
        "0000000000020000"
    ))).reshape(15, 4, 4)[rounds]

    tweakeys = np.array(pad(bytearray.fromhex(
        "0000080D00000800" "0000040800000500" "00000E0D00000C00"
        "000800000000080D" "000B000000000408" "000E000000000E0D"
        "0D08000000080000" "01090000000B0000" "060F0000000E0000"
        "000000080D080000" "0000000701090000" "0000000F060F0000"
        "D000000800000008" "2000000300000007" "300000070000000F"
        "08000000D0000008" "0F00000020000003" "0700000030000007"
        "08D0000008000000" "064000000F000000" "0B90000007000000"
        "8000000008D00000" "E000000006400000" "B00000000B900000"
        "8000D00080000000" "D0009000E0000000" "50004000B0000000"
        "008000008000D000" "00C00000D0009000" "0050000050004000"
        "008000D000800000" "00A0003000C00000" "00A0002000500000"
        "00008000008000D0" "0000800000A00030" "0000A00000A00020"
        "00008D0000008000" "0000560000008000" "0000D1000000A000"
        "0000008000008D00" "0000001000005600" "000000D00000D100"
        "000D008000000080" "000D00B000000010" "00080060000000D0"
    ))).reshape(15, 3, 4, 4)[rounds]


    char = Skinny64Characteristic(sbox_in, sbox_out, tweakeys)
    cipher = Skinny64(char)
    assert char.num_rounds == numrounds
    model = cipher.solve(seed=2006)
    sbox_in = model.sbox_in #type: ignore
    sbox_out = model.sbox_out #type: ignore
    round_tweakeys = model._round_tweakeys #type: ignore

    key, tk2, tk3 = model.key, model.tk2, model.tk3 #type: ignore

    pt1 = matrix_to_packed_uint8(sbox_in[0])
    ct1 = matrix_to_packed_uint8(sbox_in[-1])
    tweakey1 = np.array([matrix_to_packed_uint8(key), matrix_to_packed_uint8(tk2), matrix_to_packed_uint8(tk3)]).flatten()


    pt2 = pt1 ^ matrix_to_packed_uint8(char.sbox_in[0])
    last_tk_delta = np.bitwise_xor.reduce(tweakeys[-1], axis=0) & tweakey_mask
    ct_delta = do_mix_cols(do_shift_rows(char.sbox_out[-1] ^ last_tk_delta))
    ct2 = ct1 ^ matrix_to_packed_uint8(ct_delta)

    tweakey2 = tweakey1 ^ np.array([matrix_to_packed_uint8(char.tweakeys[0][0]), matrix_to_packed_uint8(char.tweakeys[0][1]), matrix_to_packed_uint8(char.tweakeys[0][2])]).flatten()

    sbox = cipher.sbox

    assert np.all(sbox[sbox_in[:-1]] == sbox_out)
    assert np.all(sbox[sbox_in[:-1] ^ char.sbox_in] == sbox_out ^ char.sbox_out)

    np.set_printoptions(formatter={'int': lambda x: f'{x:02x}'})

    # sanity check result
    rtk = np.array([key, tk2, tk3])
    for i in range(numrounds):
        sbi = sbox_in[i]
        sbo = sbox_out[i]
        assert np.all(sbo == sbox[sbi])
        assert np.all(sbo ^ char.sbox_out[i] == sbox[sbi ^ char.sbox_in[i]])

        actual_rtk = round_tweakeys[i]
        print(f"round {i}".center(80, '-'))
        print(rtk.shape, actual_rtk.shape)
        print(rtk ^ actual_rtk)
        assert np.all(rtk == actual_rtk)
        rtk = update_tweakey(rtk, block_size=64)

    ct1_ref = np.array(bytearray(skinny64_enc_ecb(pt1, tweakey1, numrounds)))
    ct2_ref = np.array(bytearray(skinny64_enc_ecb(pt2, tweakey2, numrounds)))

    print(f"pt diff:           {pt1 ^ pt2}")
    print(f"tweakey diff:      {tweakey1 ^ tweakey2}")
    print(f"actual ct diff:    {ct1_ref ^ ct2_ref}")

    assert np.all(ct1_ref == ct1)
    assert np.all(ct2_ref == ct2)


def test_zero_characteristic_long_key():
    numrounds = 7
    sbox_in_delta = sbox_out_delta = np.zeros((numrounds, 4, 4), dtype=np.uint8)
    tweakeys = np.zeros((numrounds, 3, 4, 4), dtype=np.uint8)

    char = Skinny64Characteristic(sbox_in_delta, sbox_out_delta, tweakeys)
    cipher = Skinny64LongKey(char)

    model = cipher.solve(seed=9789)
    sbox_in = model.sbox_in #type: ignore
    sbox_out = model.sbox_out #type: ignore
    round_tweakeys = model.round_tweakeys #type: ignore

    pt = sbox_in[0]
    ct = sbox_in[-1]

    sbox_in = sbox_in[:-1]
    assert np.all(cipher.sbox[sbox_in] == sbox_out)

    state = pt
    for rnd in range(numrounds):
        state = cipher.sbox[state]

        state ^= expanded_rc[rnd]
        state[:2] ^= round_tweakeys[rnd]
        state = do_mix_cols(do_shift_rows(state))

    assert np.all(ct == state)

@pytest.mark.parametrize("rounds", [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 10)])
def test_nonzero_characteristic_long_key(rounds):
    numrounds = len(range(rounds.start, rounds.stop))
    sbox_in_delta = np.array(pad(bytearray.fromhex(
        "0000000140000004"
        "0000000000000020"
        "010D000D0000000D"
        "0020000020000000"
        "0000003000300000"
        "0000C000000C0000"
        "0200000000000200"
        "3000000000000000"
        "0000000000000000"
        "0000000000000000"
        "0010001000000010"
        "0A00000000050000"
        "00000A0000000000"
        "0000000000000000"
        "0000000000040000"
    ))).reshape(15, 4, 4)[rounds]

    sbox_out_delta = np.array(pad(bytearray.fromhex(
        "0000000820000002"
        "0000000000000010"
        "0A0E000200000002"
        "0030000030000000"
        "000000C000C00000"
        "0000200000020000"
        "0500000000000300"
        "D000000000000000"
        "0000000000000000"
        "0000000000000000"
        "00800090000000A0"
        "0A000000000A0000"
        "00000A0000000000"
        "0000000000000000"
        "0000000000020000"
    ))).reshape(15, 4, 4)[rounds]

    tweakeys = np.array(pad(bytearray.fromhex(
        "0000080D00000800" "0000040800000500" "00000E0D00000C00"
        "000800000000080D" "000B000000000408" "000E000000000E0D"
        "0D08000000080000" "01090000000B0000" "060F0000000E0000"
        "000000080D080000" "0000000701090000" "0000000F060F0000"
        "D000000800000008" "2000000300000007" "300000070000000F"
        "08000000D0000008" "0F00000020000003" "0700000030000007"
        "08D0000008000000" "064000000F000000" "0B90000007000000"
        "8000000008D00000" "E000000006400000" "B00000000B900000"
        "8000D00080000000" "D0009000E0000000" "50004000B0000000"
        "008000008000D000" "00C00000D0009000" "0050000050004000"
        "008000D000800000" "00A0003000C00000" "00A0002000500000"
        "00008000008000D0" "0000800000A00030" "0000A00000A00020"
        "00008D0000008000" "0000560000008000" "0000D1000000A000"
        "0000008000008D00" "0000001000005600" "000000D00000D100"
        "000D008000000080" "000D00B000000010" "00080060000000D0"
    ))).reshape(15, 3, 4, 4)[rounds]


    char = Skinny64Characteristic(sbox_in_delta, sbox_out_delta, tweakeys)
    cipher = Skinny64LongKey(char)

    model = cipher.solve(seed=9789)
    sbox_in = model.sbox_in #type: ignore
    sbox_out = model.sbox_out #type: ignore
    round_tweakeys = model.round_tweakeys #type: ignore

    pt = sbox_in[0]
    ct = sbox_in[-1]

    sbox_in = sbox_in[:-1]
    assert np.all(cipher.sbox[sbox_in] == sbox_out)

    state = pt
    for rnd in range(numrounds):
        state = cipher.sbox[state]

        state ^= expanded_rc[rnd]
        state[:2] ^= round_tweakeys[rnd]
        state = do_mix_cols(do_shift_rows(state))

    assert np.all(ct == state)
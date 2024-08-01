from __future__ import annotations

from random import randint
from typing import Any

import numpy as np

from autodiver.ascon.ascon_model import Ascon, AsconCharacteristic
from autodiver.cipher_model import count_solutions

from autodiver_ciphers.ascon.pyascon import ascon_permutation
from shutil import which

import pytest
from icecream import ic


approxmc = which("approxmc")


def rotr(val: np.uint64, r: int):
    val_int = int(val)
    res_int = (val_int >> r) | ((val_int & (1<<r)-1) << (64-r))
    return np.uint64(res_int)


def ascon_inv_linear_layer(inp: np.ndarray[Any, np.dtype[np.uint64]]) -> np.ndarray[Any, np.dtype[np.uint64]]:
    rotations = [
        [0, 3, 6, 9, 11, 12, 14, 15, 17, 18, 19, 21, 22, 24, 25, 27, 30, 33, 36, 38, 39, 41, 42, 44, 45, 47, 50, 53, 57, 60, 63],
        [0, 1, 2, 3, 4, 8, 11, 13, 14, 16, 19, 21, 23, 24, 25, 27, 28, 29, 30, 35, 39, 43, 44, 45, 47, 48, 51, 53, 54, 55, 57, 60, 61],
        [0, 2, 4, 6, 7, 10, 11, 13, 14, 15, 17, 18, 20, 23, 26, 27, 28, 32, 34, 35, 36, 37, 40, 42, 46, 47, 52, 58, 59, 60, 61, 62, 63],
        [1, 2, 4, 6, 7, 9, 12, 17, 18, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 40, 42, 44, 47, 48, 49, 53, 58, 61, 63],
        [0, 1, 2, 3, 4, 5, 9, 10, 11, 13, 16, 20, 21, 22, 24, 25, 28, 29, 30, 31, 35, 36, 40, 41, 44, 45, 46, 47, 48, 50, 53, 55, 60, 61, 63],
    ]
    rotations = [np.array(rot, dtype=np.uint64) for rot in rotations]

    output = np.zeros_like(inp)
    for i in range(5):
        for rot in rotations[i]:
            output[i] ^= (inp[i] >> rot) | (inp[i] << (np.uint64(64) - rot))

    return output

def ascon_linear_layer(inp: np.ndarray[Any, np.dtype[np.uint64]]) -> np.ndarray[Any, np.dtype[np.uint64]]:
    S = inp.copy()

    S[0] ^= rotr(S[0], 19) ^ rotr(S[0], 28)
    S[1] ^= rotr(S[1], 61) ^ rotr(S[1], 39)
    S[2] ^= rotr(S[2],  1) ^ rotr(S[2],  6)
    S[3] ^= rotr(S[3], 10) ^ rotr(S[3], 17)
    S[4] ^= rotr(S[4],  7) ^ rotr(S[4], 41)

    return S

def ascon_sbox(sbi: np.ndarray[Any, np.dtype[np.uint64]]) -> np.ndarray[Any, np.dtype[np.uint64]]:
    assert sbi.shape == (5,)

    S = sbi.copy()
    T = np.zeros_like(S)


    S[0] ^= S[4]
    S[4] ^= S[3]
    S[2] ^= S[1]
    T = [(S[i] ^ 0xFFFFFFFFFFFFFFFF) & S[(i+1)%5] for i in range(5)]
    for i in range(5):
        S[i] ^= T[(i+1)%5]
    S[1] ^= S[0]
    S[0] ^= S[4]
    S[3] ^= S[2]
    S[2] ^= 0XFFFFFFFFFFFFFFFF

    return S


@pytest.mark.skipif(approxmc is None, reason="approxmc not found")
def test_zero_characteristic():
    numrounds = 5
    sbi = sbo = np.zeros((numrounds, 5), dtype=np.uint64)

    char = AsconCharacteristic(sbi, sbo)
    ascon = Ascon(char)

    ic(ascon.sbox_in[0])
    ic(ascon.sbox_in_bitsliced[0])

    model = ascon.solve(seed=6019)

    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    sbi_bitsliced = model._actual_sbox_in[:64*numrounds]# type: ignore
    sbo_bitsliced = model._actual_sbox_out # type: ignore

    ic(sbi_bitsliced[0])
    ic(sbo_bitsliced[0])

    constants = np.array([0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b], dtype=np.uint8)

    ic(ascon.sbox[sbi_bitsliced] ^ sbo_bitsliced)
    assert np.all(ascon.sbox[sbi_bitsliced[0]] == sbo_bitsliced[0])

    for i in range(numrounds):
        this_sbi = sbi[i].copy()
        this_sbi[2] ^= constants[12 - numrounds + i]

        this_sbo = sbo[i]
        ref_sbo = ascon_sbox(this_sbi.copy())

        ic(this_sbo)
        ic(ref_sbo)

        ic(this_sbo ^ ref_sbo)
        assert np.all(this_sbo == ref_sbo)

    # assert np.all(ascon_sbox(sbi[0]) == sbo[0])



    numrounds = 2
    sbi = sbo = np.zeros((numrounds, 5), dtype=np.uint64)

    char = AsconCharacteristic(sbi, sbo)
    ascon = Ascon(char)
    num_solutions = count_solutions(ascon.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (320)

    # for bit_var in gift.key.flatten():
    #     gift.cnf += CNF([bit_var * (-1)**randint(0,1), 0])

    # num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1 << 64

    # for bit_var in gift.sbox_in[0].flatten():
    #     gift.cnf += CNF([bit_var * (-1)**randint(0,1), 0])

    # model = gift.solve(seed=9334)

    # key = model.key # type: ignore
    # sbi = model.sbox_in # type: ignore
    # sbo = model.sbox_out # type: ignore

    # assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)

    # for r, round_sbi in enumerate(sbi):
    #     ref = gift64_enc(sbi[0], key, r)
    #     assert np.all(round_sbi == ref)

    # num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1

def test_nonzero_characteristic():
    np.set_printoptions(formatter={'int': lambda x: f"{x:016x}"})
    sbi_delta = np.array(bytearray.fromhex(
        "0000000000000000" "8000000000000000" "8000000000000000" "0000000000000000" "0000000000000000"
        "8000100800000000" "0000000000000000" "0000000000000000" "0000000000000000" "0000000000000000"
        "8000000002000080" "9000904801000024" "0000000000000000" "0000000000000000" "0000000000000000"
        "0000000000000000" "0004000112000480" "4a40da2d21840036" "1916986c5b2440a4" "0001000042040081")
    ).reshape(4, 40).view(np.uint64).byteswap()

    sbo_delta = np.array([ascon_inv_linear_layer(round_sbi) for round_sbi in sbi_delta[1:]])

    for i in range(len(sbi_delta) - 1):
        assert np.all(ascon_linear_layer(sbo_delta[i]) == sbi_delta[i + 1])

    for i in range(4):
        print(i)
        print(" ".join(f"{x:016x}" for x in sbi_delta[i]).replace("0", "."))
        if i in range(len(sbo_delta)):
            print(" ".join(f"{x:016x}" for x in sbo_delta[i]).replace("0", "."))
        print()


    output_delta = sbi_delta[-1]
    sbi_delta = sbi_delta[:-1]

    assert sbi_delta.shape == sbo_delta.shape

    char = AsconCharacteristic(sbi_delta, sbo_delta)
    ic(char.num_rounds)
    ascon = Ascon(char)


    assert np.all((char.sbox_in != 0) == (char.sbox_out != 0))

    ddt_probs = Ascon.ddt[char.sbox_in, char.sbox_out]

    assert np.all(ddt_probs > 0)
    assert np.log2(ddt_probs / 32).sum() == -40

    model = ascon.solve(seed=8850)

    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore

    actual_sbi = model._actual_sbox_in.reshape(char.num_rounds, 64) # type: ignore
    actual_sbo = model._actual_sbox_out.reshape(char.num_rounds, 64) # type: ignore

    assert np.all(ascon.sbox[actual_sbi] == actual_sbo)
    assert np.all(ascon.sbox[actual_sbi ^ char.sbox_in] == actual_sbo ^ char.sbox_out)

    idx = np.where(char.sbox_in != 0)
    ic(actual_sbi.shape)
    ic(actual_sbi[idx])
    ic(actual_sbo[idx])


    assert sbi.shape == sbi_delta.shape
    assert sbo.shape == sbo_delta.shape

    assert np.all(ascon.sbox_in > 0)
    assert np.all(ascon.sbox_out > 0)

    # test linear layer
    for r in range(char.num_rounds - 1):
        assert np.all(ascon_linear_layer(sbo[r]) == sbi[r+1])

    for r in range(1, char.num_rounds + 1):
        ic("")
        ic(r)
        ascon_input = sbi[char.num_rounds - r].tolist()
        ascon_permutation(ascon_input, r)

        shifted_input = (sbi[char.num_rounds - r] ^ ic(sbi_delta[char.num_rounds - r])).tolist()
        ascon_permutation(shifted_input, r)

        ref_output = np.array(ascon_input, dtype=np.uint64)
        shifted_output = np.array(shifted_input, dtype=np.uint64)

        act_output = ascon_linear_layer(sbo[-1])

        assert np.all(act_output == ref_output)
        actual_output_delta = act_output ^ shifted_output
        ic(actual_output_delta)
        ic(output_delta)
        assert np.all(actual_output_delta == output_delta)

    a = sbi[0]
    b = sbi[0]

    act_output = ascon_linear_layer(sbo[char.num_rounds - 1])

    ref_output = sbi[0].copy().tolist()
    ascon_permutation(ref_output, char.num_rounds)
    ref_output = np.array(ref_output, dtype=np.uint64)

    assert np.all(act_output == ref_output)

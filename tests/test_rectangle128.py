from random import randint
import numpy as np
import pytest
from shutil import which

from autodiver.cipher_model import count_solutions
from autodiver.rectangle128.rectangle_model import Rectangle128, RectangleCharacteristic
from autodiver_ciphers.rectangle128.rectangle_cipher import rectangle_enc, nibble_to_block, nibble_to_key
from sat_toolkit.formula import CNF


def test_zero_characteristic():
    numrounds = 4
    sbi_delta = sbo_delta = np.zeros((numrounds, 16), dtype=np.uint8)
    char = RectangleCharacteristic(sbi_delta, sbo_delta)
    rectangle = Rectangle128(char)

    for bit_var in rectangle.key.flatten():
        rectangle.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    num_solutions = count_solutions(rectangle.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << 64
    for bit_var in rectangle.sbox_in[0].flatten():
        rectangle.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    model = rectangle.solve(seed=6024)

    key = model.key # type: ignore
    key = nibble_to_key(key)
    pt = model.pt # type: ignore
    pt = nibble_to_block(pt)
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(rectangle.sbox[sbi[:rectangle.num_rounds]] == sbo)

    for r in range(1, numrounds):
        out = nibble_to_block(sbi[r])
        ref = rectangle_enc(pt, key[0], key[1], r)
        print(f"{r = }")
        print(f'{pt   = :016x}')
        # print(f'{out  = :016x}')
        # print(f'{ref  = :016x}')
        print(f'diff = {out ^ ref:016x}')
        print()
        assert out == ref
    # num_solutions = count_solutions(rectangle.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1


def test_nonzero_characteristic():
    # characteristic from https://eprint.iacr.org/2022/837 (Table 11)
    # char = (
    #     ("0020000600000000", "0060000200000000"),
    #     ("0200006000000000", "0600002000000000"),
    #     ("2000060000000000", "6000020000000000"),
    # )

    char = (
        ("0000000060000200", "0000000020000600"),
        ("0000000006000020", "0000000002000060"),
        ("0000000000600002", "0000000000200006"),
    )

    sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
    sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
    # sbi_delta = sbi_delta[:, ::-1]
    # sbo_delta = sbo_delta[:, ::-1]

    char = RectangleCharacteristic(sbi_delta, sbo_delta)
    rectangle = Rectangle128(char)
    model = rectangle.solve(seed=8284)

    key = model.key # type: ignore
    key = nibble_to_key(key)
    pt = model.pt # type: ignore
    pt = nibble_to_block(pt)
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(rectangle.sbox[sbi[:rectangle.num_rounds]] == sbo)

    delta_0 = nibble_to_block(sbi_delta[0])
    for r in range(1, char.num_rounds):
        out = nibble_to_block(sbi[r])
        ref = rectangle_enc(pt, key[0], key[1], r)
        ref_xor = rectangle_enc(pt ^ delta_0, key[0], key[1], r)
        print(f"{r = }")
        print(f'diff = {out ^ ref:016x}')
        assert out == ref
        expected_diff = nibble_to_block(sbi_delta[r])
        print(f"{ref ^ ref_xor = :016x}")
        print(f"{expected_diff = :016x}")
        print()
        assert expected_diff == ref ^ ref_xor

if __name__ == "__main__":
    test_zero_characteristic()
    test_nonzero_characteristic()

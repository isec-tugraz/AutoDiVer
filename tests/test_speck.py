from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from autodiver.speck.speck_model import Speck32LongKey, Speck64LongKey, Speck128LongKey, SpeckCharacteristic
from autodiver.speck.speck_util import rotr_speck, rotl_speck

class SpeckLongKey():
    def __init__(self, wordsize: int):
        self.wordsize = wordsize

    def enc(self, pt: np.ndarray[Any, np.dtype[np.uint64]], round_key: np.ndarray[Any, np.dtype[np.uint64]]) -> np.ndarray[Any, np.dtype[np.uint64]]:
        num_rounds = len(round_key)
        l, r = pt
        for rnd in range(num_rounds):
            key = round_key[rnd]
            l = rotr_speck(l, self.wordsize)
            l = (int(l) + int(r)) & (2**self.wordsize - 1)
            l ^= key
            r = rotl_speck(r, self.wordsize)
            r ^= l
        return np.array([l, r], np.uint64)


@pytest.mark.parametrize('wordsize', [16, 32, 64])
def test_zero_char(wordsize):
    numrounds = 20
    input_diffs = np.zeros((numrounds + 1, 2), np.uint64)

    char = SpeckCharacteristic(input_diffs, wordsize=wordsize, file_path=None)

    if wordsize == 16:
        cipher = Speck32LongKey(char)
    elif wordsize == 32:
        cipher = Speck64LongKey(char)
    elif wordsize == 64:
        cipher = Speck128LongKey(char)
    else:
        raise ValueError(f'wordsize {wordsize} not supported')

    model = cipher.solve()
    round_inputs = np.array(model.round_in, dtype=np.uint64) # type: ignore
    round_keys = np.array(model.round_key, dtype=np.uint64) # type: ignore

    np.set_printoptions(formatter={'int': lambda x: f'{x:08x}'})

    ref_cipher = SpeckLongKey(wordsize=wordsize)
    for rnd in range(1, numrounds):
        ref_l, ref_r = ref_cipher.enc(round_inputs[0], round_keys[:rnd])
        assert ref_l == round_inputs[rnd, 0]
        assert ref_r == round_inputs[rnd, 1]


def test_ALLW15_char():
    input_diffs = np.array([
        [0x0a60, 0x4205],
        [0x0211, 0x0a04],
        [0x2800, 0x0010],
        [0x0040, 0x0000],
        [0x8000, 0x8000],
        [0x8100, 0x8102],
        [0x8000, 0x840a],
        [0x850a, 0x9520],
        [0x802a, 0xd4a8],
        [0x81a8, 0xd30b],
    ], dtype=np.uint64)

    np.set_printoptions(formatter={'int': lambda x: f'{x:04x}'})

    char = SpeckCharacteristic(input_diffs, wordsize=16, file_path=None)
    # from IPython import embed; embed()
    cipher = Speck32LongKey(char)


    model = cipher.solve()
    round_inputs = np.array(model.round_in, dtype=np.uint64) # type: ignore
    round_keys = np.array(model.round_key, dtype=np.uint64) # type: ignore

    ref_cipher = SpeckLongKey(wordsize=16)
    for rnd in range(1, cipher.num_rounds):
        ref_l, ref_r = ref_cipher.enc(round_inputs[0], round_keys[:rnd])
        ref2_l, ref2_r = ref_cipher.enc(round_inputs[0] ^ char.round_in[0], round_keys[:rnd])

        assert ref_l == round_inputs[rnd, 0]
        assert ref_r == round_inputs[rnd, 1]

        expected_diff = char.round_in[rnd]
        observed_diff = np.array([ref_l ^ ref2_l, ref_r ^ ref2_r], np.uint64)

        print(f'Round {rnd}')
        print(f'observed_diff: {observed_diff}')
        print(f'expected_diff: {expected_diff}')
        assert np.all(observed_diff == expected_diff)

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autodiver.speck.speck_characteristic import SpeckCharacteristic
from autodiver.speck.speck_model import _SpeckBase, Speck32LongKey, Speck48LongKey, Speck64LongKey, Speck96LongKey, Speck128LongKey, Speck32Characteristic, Speck48Characteristic, Speck64Characteristic, Speck96Characteristic, Speck128Characteristic
from autodiver.speck.speck_util import rotr_speck, rotl_speck

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
TRAILS_DIR: Path = REPO_ROOT / "trails" / "speck"

# wordsize -> (characteristic class, long-key cipher class)
SPECK_VARIANTS: dict[int, tuple[type[SpeckCharacteristic], type[_SpeckBase]]] = {
    16: (Speck32Characteristic, Speck32LongKey),
    24: (Speck48Characteristic, Speck48LongKey),
    32: (Speck64Characteristic, Speck64LongKey),
    48: (Speck96Characteristic, Speck96LongKey),
    64: (Speck128Characteristic, Speck128LongKey),
}

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


@pytest.mark.parametrize('wordsize', [16, 24, 32, 48, 64])
def test_zero_char(wordsize):
    numrounds = 20
    input_diffs = np.zeros((numrounds + 1, 2), np.uint64)

    #char = SpeckCharacteristic(input_diffs, wordsize=wordsize, file_path=None)

    if wordsize == 16:
        char = Speck32Characteristic(input_diffs, file_path=None)
        cipher = Speck32LongKey(char)
    elif wordsize == 24:
        char = Speck48Characteristic(input_diffs, file_path=None)
        cipher = Speck48LongKey(char)
    elif wordsize == 32:
        char = Speck64Characteristic(input_diffs, file_path=None)
        cipher = Speck64LongKey(char)
    elif wordsize == 48:
        char = Speck96Characteristic(input_diffs, file_path=None)
        cipher = Speck96LongKey(char)
    elif wordsize == 64:
        char = Speck128Characteristic(input_diffs, file_path=None)
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


SPECK_CHARS: list[tuple[int, Path, tuple[int, int]|None]] = [
    (16, TRAILS_DIR / "speck32_r6_BR22_table_18a.npz", None),
    # (16, TRAILS_DIR / "speck32_r7_BR22_table_18b.npz", None),
    # (16, TRAILS_DIR / "speck32_r8_BR22_table_19a.npz", None),
    # (16, TRAILS_DIR / "speck32_r8_BR22_table_19b.npz", None),
    (16, TRAILS_DIR / "speck32_r9_ALLW14_table_7.npz", None),
    # (16, TRAILS_DIR / "speck32_r9_BR22_table_20a.npz", None),
    # (16, TRAILS_DIR / "speck32_r9_BR22_table_20b.npz", None),
    # (16, TRAILS_DIR / "speck32_r9_BR22_table_20c.npz", None),
    (24, TRAILS_DIR / "speck48_r10_ALLW14_table_7_fixed.npz", None),
    # (24, TRAILS_DIR / "speck48_r11_BR22_table_22_1a.npz", None),
    # (24, TRAILS_DIR / "speck48_r11_BR22_table_22_1b.npz", None),
    (32, TRAILS_DIR / "speck64_r13_ALLW14_table_9.npz", None),
    # (32, TRAILS_DIR / "speck64_r15_BR22_table_22b.npz", None),
    # (48, TRAILS_DIR / "speck96_r15_BR22_table_22c.npz", None),
    (64, TRAILS_DIR / "speck128_r20_BR22_table_23a.npz", None),
    # (64, TRAILS_DIR / "speck128_r20_BR22_table_23b.npz", None),
    # (64, TRAILS_DIR / "speck128_r20_BR22_table_23c.npz", None),
    # (64, TRAILS_DIR / "speck128_r20_BR22_table_23d.npz", None),
]


@pytest.mark.parametrize('wordsize,characteristic_path,truncate_rounds',
                         SPECK_CHARS,
                         ids=[path.stem for _, path, _ in SPECK_CHARS])
def test_nonzero_char(wordsize: int, characteristic_path: Path, truncate_rounds: tuple[int, int]|None):
    """check that the SAT model reproduces the published differential characteristic"""
    np.set_printoptions(formatter={'int': lambda x: f'{x:04x}'})

    characteristic_cls, cipher_cls = SPECK_VARIANTS[wordsize]

    char = characteristic_cls.load(characteristic_path)
    if truncate_rounds is not None:
        char.truncate_rounds(truncate_rounds)
    cipher = cipher_cls(char)

    model = cipher.solve()
    round_inputs = np.array(model.round_in, dtype=np.uint64) # type: ignore
    round_keys = np.array(model.round_key, dtype=np.uint64) # type: ignore

    ref_cipher = SpeckLongKey(wordsize=wordsize)
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

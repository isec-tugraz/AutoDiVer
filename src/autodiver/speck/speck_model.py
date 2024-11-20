from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sat_toolkit.formula import XorCNF

from autodiver.arx_util import model_full_adder, model_modular_addition
from autodiver.cipher_model import SboxCipher, DifferentialCharacteristic
from .speck_util import rotr_speck_np, ALPHA_MAP, BETA_MAP


class SpeckCharacteristic(DifferentialCharacteristic):
    round_in: np.ndarray[Any, np.dtype[np.uint64]]
    add_in1: np.ndarray[Any, np.dtype[np.uint64]]
    add_in2: np.ndarray[Any, np.dtype[np.uint64]]
    add_out: np.ndarray[Any, np.dtype[np.uint64]]

    def __init__(self, round_in: np.ndarray, wordsize: int, file_path: Path|None):
        self.rounds_from_to = None
        self.file_path = file_path
        self.round_in = round_in
        self.num_rounds = len(round_in) - 1

        assert self.round_in.shape == (self.num_rounds + 1, 2)

        self.add_in1 = rotr_speck_np(round_in[:-1, 0], wordsize)
        self.add_in2 = round_in[:-1, 1]
        self.add_out = round_in[1:, 0]

        self.wordsize = wordsize

    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            wordsize = int(f['wordsize'])
            round_in = np.array(f['round_in'], dtype=np.uint64)

        return cls(round_in=round_in, wordsize=wordsize, file_path=characteristic_path)

    def truncate_rounds(self, rounds_from_to: tuple[int, int]):
        current_rounds = range(self.num_rounds)
        start, end = rounds_from_to

        assert start in current_rounds
        assert end in current_rounds

        self.round_in = self.round_in[start:end + 2]
        self.add_in1 = self.add_in1[start:end + 1]
        self.add_in2 = self.add_in2[start:end + 1]
        self.add_out = self.add_out[start:end + 1]
        self.rounds_from_to = rounds_from_to

        self.num_rounds = end + 1 - start
        assert self.num_rounds == len(self.add_in1) == len(self.add_in2) == len(self.add_out) == len(self.round_in) - 1

    def log2_ddt_probability(self):
        # TODO: implement
        return float('nan')


class _SpeckBase(SboxCipher):
    num_rounds: int
    wordsize: int
    add_in1: np.ndarray[Any, np.dtype[np.int32]]
    add_in2: np.ndarray[Any, np.dtype[np.int32]]
    add_out: np.ndarray[Any, np.dtype[np.int32]]
    _carry: np.ndarray[Any, np.dtype[np.int32]]

    adder_assumptions: np.ndarray[Any, np.dtype[np.int32]]

    round_input: np.ndarray[Any, np.dtype[np.int32]]
    round_key: np.ndarray[Any, np.dtype[np.int32]]
    ct: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: SpeckCharacteristic, **kwargs):
        super().__init__(char, **kwargs)


        if char.wordsize != self.wordsize:
            raise ValueError(f'wordsize of characteristic ({char.wordsize}) does not match wordsize of cipher ({self.wordsize})')
        self.char = char

        assert self.block_size == 2 * self.wordsize

        self.add_index_array('round_key', (self.num_rounds, self.wordsize))
        self.add_index_array('add_in1', (self.num_rounds, self.wordsize))
        self.add_index_array('add_in2', (self.num_rounds, self.wordsize))
        self.add_index_array('add_out', (self.num_rounds, self.wordsize))
        self.add_index_array('_carry', (self.num_rounds, self.wordsize))
        self.add_index_array('ct', (2, self.wordsize))

        self.key = self.round_key
        self._fieldnames.add('key')
        self.add_index_array('tweak', (0,))
        self.key_size = self.num_rounds * self.wordsize

        self.round_in = np.zeros((self.num_rounds + 1, 2, self.wordsize), dtype=np.uint64)
        self.round_in[:-1, 0] = np.roll(self.add_in1, ALPHA_MAP[self.wordsize], axis=1)
        self.round_in[:-1, 1] = self.add_in2
        self.round_in[-1] = self.ct
        self._fieldnames.add('round_in')

        self.pt = self.round_in[0]
        self.ct = self.round_in[-1]
        self._fieldnames.add('pt')
        self._fieldnames.add('ct')

        if self.model_sbox_assumptions:
            self.add_index_array("adder_assumptions", (self.num_rounds, self.wordsize))
        else:
            self.add_index_array("adder_assumptions", (0,))
        self.sbox_assumptions = self.adder_assumptions

        self._model_addition()
        self._model_linear_layer()

    def _model_addition(self):
        for r in range(self.num_rounds):
            in_delta = (self.char.add_in1[r], self.char.add_in2[r])

            model = model_modular_addition(in_delta, self.char.add_out[r], self.wordsize, self.model_sbox_assumptions)

            add_in1_vars = self.add_in1[r].tolist()
            add_in2_vars = self.add_in2[r].tolist()
            add_out_vars = self.add_out[r].tolist()
            carry_vars = self._carry[r].tolist()
            assumption_vars = self.adder_assumptions[r].tolist() if self.model_sbox_assumptions else []

            new_vars = np.array([0] + add_in1_vars + add_in2_vars + add_out_vars + carry_vars + assumption_vars, dtype=np.int32)
            self.cnf += model.translate(new_vars)


    def _model_linear_layer(self):
        for r in range(self.num_rounds):
            rotated_inp2 = np.roll(self.add_in2[r], BETA_MAP[self.wordsize])
            add_out = self.add_out[r]
            key = self.round_key[r]

            round_out1 = self.round_in[r + 1, 0]
            round_out2 = self.round_in[r + 1, 1]

            self.cnf += XorCNF.create_xor(add_out, key, round_out1)
            self.cnf += XorCNF.create_xor(round_out1, rotated_inp2, round_out2)

    @classmethod
    def _fmt_arr(cls, arr: np.ndarray, cellsize: int):
        print(arr.shape, arr.dtype, cellsize)
        if cellsize == 0 and len(arr) == 0:
            return ''
        if cellsize != cls.wordsize:
            raise ValueError(f'cellsize must be {cls.wordsize} not {cellsize}')
        return ' '.join(f'{x:0{cellsize // 4}x}' for x in arr)


class Speck32LongKey(_SpeckBase):
    cipher_name = "Speck32LongKey"
    sbox = None # type: ignore
    ddt = None # type: ignore
    wordsize = 16
    block_size = 32
    key_size: int
    tweak_size = 0

class Speck48LongKey(_SpeckBase):
    cipher_name = "Speck48LongKey"
    sbox = None # type: ignore
    ddt = None # type: ignore
    wordsize = 24
    block_size = 48
    key_size: int
    tweak_size = 0

class Speck64LongKey(_SpeckBase):
    cipher_name = "Speck64LongKey"
    sbox = None # type: ignore
    ddt = None # type: ignore
    wordsize = 32
    block_size = 64
    key_size: int
    tweak_size = 0

class Speck96LongKey(_SpeckBase):
    cipher_name = "Speck96LongKey"
    sbox = None # type: ignore
    ddt = None # type: ignore
    wordsize = 48
    block_size = 96
    key_size: int
    tweak_size = 0

class Speck128LongKey(_SpeckBase):
    cipher_name = "Speck128LongKey"
    sbox = None # type: ignore
    ddt = None # type: ignore
    wordsize = 64
    block_size = 128
    key_size: int
    tweak_size = 0

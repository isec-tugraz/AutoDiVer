from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sat_toolkit.formula import XorCNF, CNF

from autodiver.arx_util import model_full_adder, model_modular_addition
from autodiver.cipher_model import SboxCipher, DifferentialCharacteristic
from autodiver.arx_util import modular_addition_probability
from .speck_util import rotr_speck_np, ALPHA_MAP, BETA_MAP
from pysat.card import CardEnc, IDPool

from .speck_characteristic import SpeckCharacteristic, Speck32Characteristic, Speck48Characteristic, Speck64Characteristic, Speck96Characteristic, Speck128Characteristic

class _SpeckBase(SboxCipher):
    num_rounds: int
    wordsize: int
    add_in1: np.ndarray[Any, np.dtype[np.int32]]
    add_in2: np.ndarray[Any, np.dtype[np.int32]]
    add_out: np.ndarray[Any, np.dtype[np.int32]]
    _carry: np.ndarray[Any, np.dtype[np.int32]]
    aux1: np.ndarray[Any, np.dtype[np.int32]]
    aux2: np.ndarray[Any, np.dtype[np.int32]]
    aux3: np.ndarray[Any, np.dtype[np.int32]]

    sbox_count = 2 # pro forma:)

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

        if self.search_char:
            # required for encoding of modular addition
            self.add_index_array("aux1", (self.num_rounds, self.wordsize - 1,))
            self.add_index_array("aux2", (self.num_rounds, self.wordsize - 1,))
            self.add_index_array("aux3", (self.num_rounds, self.wordsize - 1,))
            self.add_index_array("ddt_weights", (self.num_rounds, self.wordsize - 1,))
            self.add_index_array("key", (0,))
            self.key_size = 0
            if self.log_prob_boundary == None:
                self.log_prob = 0  # transitions with probability 1 possible
        else:
            self.add_index_array('_carry', (self.num_rounds, self.wordsize))
            self.add_index_array('round_key', (self.num_rounds, self.wordsize))
            self.key = self.round_key
            self._fieldnames.add('key')
            self.key_size = self.num_rounds * self.wordsize

        self.add_index_array('add_in1', (self.num_rounds, self.wordsize))
        self.add_index_array('add_in2', (self.num_rounds, self.wordsize))
        self.add_index_array('add_out', (self.num_rounds, self.wordsize))
        self.add_index_array('ct', (2, self.wordsize))

        self.add_index_array('tweak', (0,))

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

        if self.search_char:
            self._model_differential_addition()
        else:
            self._model_addition()

        self._model_linear_layer()

    def _setup_ddt(self):
        pass # no sbox here

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

    def _model_differential_addition(self):
        # encode validity condition and weights

        # index 0 : a[-1] = 0
        for r in range(self.num_rounds):

            self.cnf += XorCNF.create_xor([self.add_in1[r][0]], [self.add_in2[r][0]], [self.add_out[r][0]])

            self.cnf += XorCNF.create_xor(self.aux1[r], -self.add_in1[r][:-1], self.add_in2[r][:-1])
            self.cnf += XorCNF.create_xor(self.aux2[r], -self.add_in1[r][:-1], self.add_out[r][:-1])
            self.cnf += XorCNF.create_xor(self.aux3[r], self.add_in1[r][1:], self.add_in2[r][1:], self.add_out[r][1:], self.add_in2[r][:-1])

            reg_CNF = CNF()

            for i in range(self.wordsize - 1):
                reg_CNF += [-self.aux1[r][i], -self.aux2[r][i], -self.aux3[r][i], 0]

                # weight encoding:
                reg_CNF += [self.aux1[r][i], self.ddt_weights[r][i], 0]
                reg_CNF += [self.aux2[r][i], self.ddt_weights[r][i], 0]
                reg_CNF += [-self.ddt_weights[r][i], -self.aux1[r][i], -self.aux2[r][i], 0]

            self.cnf += reg_CNF

        self._exclude_zero_characteristic()

    def _exclude_zero_characteristic(self):
        vpool = IDPool(start_from=self.numvars + 1)
        exclude_zero_conditions = CardEnc.atleast(
            lits=self.round_in[0].flatten().tolist(), vpool=vpool, bound=1).clauses
        exclude_zero_cnf = CNF()
        for clause in exclude_zero_conditions:
            exclude_zero_cnf += clause + [0]

        self.add_index_array("exclude_zero_vars", (vpool.top - self.numvars,))
        self.cnf += exclude_zero_cnf


    def _model_linear_layer(self):
        for r in range(self.num_rounds):
            rotated_inp2 = np.roll(self.add_in2[r], BETA_MAP[self.wordsize])
            add_out = self.add_out[r]

            round_out1 = self.round_in[r + 1, 0]
            round_out2 = self.round_in[r + 1, 1]

            if self.search_char:
                self.cnf += XorCNF.create_xor(add_out, round_out1)
            else:
                key = self.round_key[r]
                self.cnf += XorCNF.create_xor(add_out, key, round_out1)

            self.cnf += XorCNF.create_xor(round_out1, rotated_inp2, round_out2)

    @classmethod
    def _fmt_arr(cls, arr: np.ndarray, cellsize: int):
        # print(arr.shape, arr.dtype, cellsize)
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

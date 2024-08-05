#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for Pyjamask96.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF

from .util import pyjamask_mix_rows_96, unload_state
from ..util import get_ddt
from ..cipher_model import SboxCipher, DifferentialCharacteristic

import scipy
from scipy.linalg import circulant

from pathlib import Path


log = logging.getLogger(__name__)


class Pyjamask96Characteristic(DifferentialCharacteristic):
    def __init__(self, sbox_in: np.ndarray[Any, np.dtype[np.uint32]], sbox_out: np.ndarray[Any, np.dtype[np.uint32]], **kwargs):
        # do reverse bit slicing
        assert sys.byteorder == 'little'
        assert sbox_in.dtype.byteorder in ['<', '=']
        assert sbox_out.dtype.byteorder in ['<', '=']
        assert sbox_in.dtype == sbox_out.dtype == np.uint32

        for i in range(len(sbox_in) - 1):
            assert np.all(pyjamask_mix_rows_96(sbox_out[i]) == sbox_in[i + 1])

        # unpack bits
        sbox_in_bits = np.unpackbits(sbox_in.view(np.uint8), axis=-1, bitorder='little').reshape(-1, 3, 32)
        sbox_out_bits = np.unpackbits(sbox_out.view(np.uint8), axis=-1, bitorder='little').reshape(-1, 3, 32)

        # bit slice
        sbox_in_bits = sbox_in_bits.swapaxes(-1, -2)
        sbox_out_bits = sbox_out_bits.swapaxes(-1, -2)

        # swap order because leftmost bit is MSB
        sbox_in_bits = sbox_in_bits[..., ::-1]
        sbox_out_bits = sbox_out_bits[..., ::-1]

        # pack bits
        sbox_in_unbitsliced = np.packbits(sbox_in_bits, axis=-1, bitorder='little')[..., 0]
        sbox_out_unbitsliced = np.packbits(sbox_out_bits, axis=-1, bitorder='little')[..., 0]

        assert np.all(Pyjamask96.ddt[sbox_in_unbitsliced, sbox_out_unbitsliced] > 0)

        super().__init__(sbox_in_unbitsliced, sbox_out_unbitsliced, **kwargs)


    @classmethod
    def load(cls, characteristic_path: Path) -> Pyjamask96Characteristic:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']

        sbox_in = np.array(sbox_in, dtype=np.uint32)
        sbox_out = np.array(sbox_out, dtype=np.uint32)

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')

        return cls(sbox_in, sbox_out, file_path=characteristic_path)

class Pyjamask96(SboxCipher):
    cipher_name = "Pyjamask96"
    sbox = np.array([1, 3, 6, 5, 2, 4, 7, 0])
    ddt = get_ddt(sbox)
    block_size = 96
    key_size = 128

    sbox_bits = 3
    sbox_count = 32

    mixing_matrices = [
        circulant([1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,0]),
        circulant([0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,0,0,1,1]),
        circulant([0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,1,1])
    ]

    # key: np.ndarray[Any, np.dtype[np.int32]]
    round_keys: np.ndarray[Any, np.dtype[np.int32]]
    ct: np.ndarray[Any, np.dtype[np.int32]] # for easier testing



    def __init__(self, char: Pyjamask96Characteristic, **kwargs):
        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape == (self.num_rounds, 32)
        assert self.char.sbox_in.dtype == self.char.sbox_out.dtype == np.uint8

        # generate variables
        # last sbox_in is ct
        self.add_index_array('sbox_in', (self.num_rounds + 1, 3, 32))
        self.add_index_array('sbox_out', (self.num_rounds, 3, 32))

        # Important: the cipher implementation also moves the input bytes around (load_state, unload_state).
        # This is currently not modeled and has to be considered in the testcases.
        # However, it does not impact the results of our queries.

        self.add_index_array('tweak', (0,)) # no tweak

        self.ct = self.sbox_in[-1]
        self._fieldnames.add('ct')

        self._model_key_schedule()

        self.add_index_array('pt', (3, 32))

        self._model_sboxes()
        self._model_linear_layer()


    def _model_key_schedule(self)-> None:
        raise NotImplementedError("Subclasses must implement _model_key_schedule")


    def _model_sboxes(self, sbox_in: None|np.ndarray=None, sbox_out: None|np.ndarray=None) -> None:
        sbox_in = sbox_in.copy() if sbox_in is not None else self.sbox_in.copy()
        sbox_out = sbox_out.copy() if sbox_out is not None else self.sbox_out.copy()

        # swap axes for bitsliced sboxes
        # swap bits to little endian (top most row is MSB)
        self.sbox_in_bitsliced = sbox_in.swapaxes(-1, -2)[..., ::-1]
        self.sbox_out_bitsliced = sbox_out.swapaxes(-1, -2)[..., ::-1]

        self._fieldnames.add('sbox_in_bitsliced')
        self._fieldnames.add('sbox_out_bitsliced')

        super()._model_sboxes(self.sbox_in_bitsliced, self.sbox_out_bitsliced)

    def _model_linear_layer(self):
        self.cnf += XorCNF.create_xor(self.pt.flatten(), self.sbox_in[0].flatten(), self.round_keys[0][:3].flatten())
        for r in range(self.num_rounds):
            self._model_single_linear_layer(r)

# just one
    def _model_single_linear_layer(self, round_idx) -> None:
        for i in range(3): # rows of state
            for j in range(32): # new bits of row (one row of matrix each)
                vars = []
                for idx, el in enumerate(self.mixing_matrices[i][j]): # one row of mixing matrix
                    if el == 1:
                        vars.append([self.sbox_out[round_idx][i][idx]])

                self.cnf += XorCNF.create_xor(*vars, [self.sbox_in[round_idx + 1][i][j]], [self.round_keys[round_idx + 1][i][j]])


    @classmethod
    def _fmt_arr(cls, arr: np.ndarray, cellsize: int):
        if cellsize == 0 and len(arr) == 0:
            return ''

        if cellsize == 4:
            return ''.join(f'{x:01x}' for x in arr.flatten())
        if cellsize == 8:
            return ''.join(f'{x:02x}' for x in arr.flatten())


        if arr.dtype == np.uint32:
            return ' '.join(f'{x:08x}' for x in arr.flatten())

        raise ValueError(f'cellsize must be 4 or 8 not {cellsize} -- (shape: {arr.shape})')


class Pyjamask_Longkey(Pyjamask96):
    def _model_key_schedule(self):
        self.add_index_array('round_keys', (self.num_rounds + 1, 3, 32))
        self.key = self.round_keys # this is a bit buggy because these are only 96 bits (instead of 128), but i think it isn't used anywhere..?
        self._fieldnames.add('key')
        self.key_size = self.key.size

class Pyjamask_with_Keyschedule(Pyjamask96):

    mixing_matrix_ks = [[0, 1, 1, 1],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0]]

    row_mixing_matrix_ks = circulant([1,0,1,0,1,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0])

    ks_state_after_mixing_top_row : np.ndarray[Any, np.dtype[np.uint32]]
    ks_state_after_mixing : np.ndarray[Any, np.dtype[np.uint32]]
    ks_state_after_row_diffusion : np.ndarray[Any, np.dtype[np.uint32]]

    def _mix_columns(self, round_idx):
        for i in range(32):
            for j in range(4):
                vars = []
                for idx, el in enumerate(self.mixing_matrix_ks[j]):
                    if(el == 1):
                        vars.append([self.round_keys[round_idx][idx][i]])

                self.cnf += XorCNF.create_xor(*vars, [self.ks_state_after_mixing[round_idx][j][i]])


    def _mix_row(self, round_idx):
        for j in range(32):
            vars = []
            for idx, el in enumerate(self.row_mixing_matrix_ks[j]):
                if el == 1:
                    vars.append([self.ks_state_after_mixing[round_idx][0][idx]])

            self.cnf += XorCNF.create_xor(*vars, [self.ks_state_after_row_diffusion[round_idx][j]])

    def _model_key_schedule_less_efficient(self):
        self.add_index_array('round_keys', (self.num_rounds + 1, 4, 32))

        self.key = self.round_keys[0]
        self._fieldnames.add('key')

        self.add_index_array('ks_state_after_mixing', (self.num_rounds, 4, 32))
        self.add_index_array('ks_state_after_row_diffusion', (self.num_rounds, 32))

        constants = np.zeros((4, 32), dtype=np.int32)
        for i in range(32):
            if (1 << i) & (0x6a << 8):
                constants[1][i] = 1

            if (1 << i) & (0x3f << 16):
                constants[2][i] = 1

            if (1 << i) & (0x24 << 24):
                constants[3][i] = 1

        for r in range(self.num_rounds):
            constants[0] = np.zeros(32, dtype=np.int32)
            for i in range(32):
                if (1 << i) & (0x8 << 4 | r):
                    constants[0][i] = 1

            self._mix_columns(r)
            self._mix_row(r)

            self.cnf += XorCNF.create_xor(self.ks_state_after_row_diffusion[r], self.round_keys[r + 1][0],
                                          rhs=constants[0])
            self.cnf += XorCNF.create_xor(np.roll(self.ks_state_after_mixing[r][1], -8), self.round_keys[r + 1][1],
                                          rhs=constants[1])
            self.cnf += XorCNF.create_xor(np.roll(self.ks_state_after_mixing[r][2], -15), self.round_keys[r + 1][2],
                                          rhs=constants[2])
            self.cnf += XorCNF.create_xor(np.roll(self.ks_state_after_mixing[r][3], -18), self.round_keys[r + 1][3],
                                          rhs=constants[3])


    def _model_key_schedule(self):
        self.add_index_array('round_keys', (self.num_rounds + 1, 4, 32))

        self.key = self.round_keys[0]
        self._fieldnames.add('key')

        self.add_index_array('ks_state_after_mixing_top_row', (self.num_rounds, 32))
        # self.add_index_array('ks_state_after_row_diffusion', (self.num_rounds, 32))

        self.ks_state_after_row_diffusion = np.zeros((self.num_rounds, 32), dtype=np.int32)
        self.ks_state_after_mixing = np.zeros((self.num_rounds, 4, 32), dtype=np.int32)

        for r in range(self.num_rounds):
            self.ks_state_after_mixing[r][0] = self.ks_state_after_mixing_top_row[r]

        constants = np.zeros((4, 32), dtype=np.int32)
        for i in range(32):
            if (1 << i) & (0x6a << 8):
                constants[1][i] = 1

            if (1 << i) & (0x3f << 16):
                constants[2][i] = 1

            if (1 << i) & (0x24 << 24):
                constants[3][i] = 1

        for r in range(self.num_rounds):
            constants[0] = np.zeros(32, dtype=np.int32)
            for i in range(32):
                if (1 << i) & (0x8 << 4 | r):
                    constants[0][i] = 1

            # optimization to save on SAT vars
            self.ks_state_after_row_diffusion[r] = self.round_keys[r + 1, 0] * np.int64(-1) ** (constants[0])
            self.ks_state_after_mixing[r][1] = np.roll(self.round_keys[r + 1, 1] * np.int64(-1) ** (constants[1]), 8)
            self.ks_state_after_mixing[r][2] = np.roll(self.round_keys[r + 1, 2] * np.int64(-1) ** (constants[2]), 15)
            self.ks_state_after_mixing[r][3] = np.roll(self.round_keys[r + 1, 3] * np.int64(-1) ** (constants[3]), 18)

            self._mix_columns(r)
            self._mix_row(r)
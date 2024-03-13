#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF
from .util import DDT, RC, do_shift_rows, mixing_mat
from ..cipher_model import SboxCipher, DifferentialCharacteristic
log = logging.getLogger(__name__)
class Midori64(SboxCipher):
    cipher_name = "MIDORI64"
    sbox = np.array([int(x, 16) for x in "cad3ebf789150246"], dtype=np.uint8)
    ddt  = DDT
    block_size = 64
    key_size = 128
    sbox_bits = 4
    sbox_count = 16
    key: np.ndarray[Any, np.dtype[np.int32]]
    mc_out: np.ndarray[Any, np.dtype[np.int32]]
    def __init__(self, char: DifferentialCharacteristic):
        super().__init__(char)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape
        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')
        for i in range(1, self.num_rounds):
            lin_input = self.char.sbox_out[i - 1]
            lin_output = self.char.sbox_in[i]
            print(f'{lin_output = }')
            temp = do_linear_layer(lin_input)
            print(f'{temp = }')
            if not np.all(temp == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')
        self._create_vars()
        self._key_schedule()
        self._model_sboxes()
        self._model_linear_layer()
        self._model_add_key()
    def _create_vars(self):
        self.add_index_array('key', (2, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('mc_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')
    def _key_schedule(self) -> None:
        keyWords = self.key.copy().reshape(2, 64)
        RK = []
        for i in range(self.num_rounds):
            rk = keyWords[i % 2]
            RK.append(rk)
            # print(f'{rk = }')
        self._round_keys = np.array(RK)
    def _addKey(self, Y, X, K, RC: np.ndarray):
        X_flat = X.flatten()
        # flip bits according to round constant
        #round constants are (may be) added only in the LSB of each nibble
        for i in range(16):
            X_flat[4*i]  *= (-1)**(RC[i] & 0x1)
        key_xor_cnf = XorCNF.create_xor(X_flat, Y.flatten(), K.flatten())
        return key_xor_cnf
    def _model_add_key(self):
        for r in range(self.num_rounds):
            self.cnf += self._addKey(self.mc_out[r], self.sbox_in[r+1], self._round_keys[r], RC[r])
    @staticmethod
    def model_mix_cols(A, B):
        mc_cnf = XorCNF()
        for c in range(4):
            colA = A[(4*c):(4*c)+4]
            colB = B[(4*c):(4*c)+4]
            for r in range(4):
                colA_red = colA[mixing_mat[r] != 0, :]
                # print(f'{colB[r]}', "===>", f'{colA_red}')
                mc_cnf += XorCNF.create_xor(colB[r], *colA_red)
        return mc_cnf
    def _model_linear_layer(self):
        for r in range(self.num_rounds):
            # print(f'{self.sbox_out[r] = }')
            mc_input = do_shift_rows(self.sbox_out[r])
            mc_output = self.mc_out[r].copy()
            # print(f'{mc_input = }')
            # print(f'{mc_output = }')
            self.cnf += self.model_mix_cols(mc_input, mc_output)
#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF
from .util import DDT, RC as DDT, RC
from .util import rotate_left, rotate_column_down, get_col, add_round_constants
from ..cipher_model import SboxCipher, DifferentialCharacteristic
log = logging.getLogger(__name__)
class Rectangle(SboxCipher):
    cipher_name = "RECTANGLE"
    sbox = np.array([int(x, 16) for x in "65CA1E79B03D8F42"], dtype=np.uint8)
    ddt  = DDT
    block_size = 64
    key_size = 128
    sbox_bits = 4
    sbox_count = 16
    key: np.ndarray[Any, np.dtype[np.int32]]
    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape
        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')
        for i in range(1, self.num_rounds):
            lin_input = self.char.sbox_out[i - 1]
            lin_output = self.char.sbox_in[i]
            # print(lin_input, lin_output)
            permuted = self.apply_perm_nibble(lin_input)
            print(permuted, lin_output)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}]->sbox_in[{i}]')
        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('key', (2*self.sbox_count, self.sbox_bits))
        self.add_index_array('s_key', (self.num_rounds, 8, self.sbox_bits))
        self.add_index_array('r_key', (self.num_rounds, 2*self.sbox_count, self.sbox_bits))
        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

        self._model_sboxes()
        self._model_key_schedule()
        self._model_linear_layer()

    def applyPerm(self, array: np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.int32]]:
        offset = [0, 1, 12, 13];
        arrayOut = array.copy()
        # print(f'{array = }')
        #for each row
        for i in range(4):
           arrayOut = rotate_column_down(arrayOut, i, offset[i])
        # print(f'{arrayOut = }')
        return arrayOut
    
    def apply_perm_nibble(self, instate):
        outstate = instate.copy()
        array = np.empty((16, 4), dtype=np.uint32)
        #first get bit state from nibble state
        # print(f"{instate = }")
        for i in range(16):
            for j in range(4):
                b = (instate[i] >> j) & 0x1
                # print(b, type(b))
                array[i][j] = b
        
        # print(array)
        array = np.array(array)
        array = self.applyPerm(array)
        # print(array)
        #Then get nibble state from bit state
        for i in range(16):
            b = 0
            for j in range(4):
                b = b | (array[i][j] << j)
            outstate[i] = b
        return outstate

    def _model_key_schedule(self) -> None:
        key_cnf = XorCNF()
        RK = []
        keyWords = self.key.copy()
        for i in range(self.num_rounds):
            array = np.empty((32, 4), dtype=np.uint32)
            row = np.empty((4, 32), dtype=np.uint32)
            rk_row = np.empty((4, 32), dtype=np.uint32)
            
            #first compute the sbox operations
            for j in range(8):
                inp = keyWords[j].reshape(-1, self.sbox_bits)[0]
                out = self.s_key[i][j].reshape(-1, self.sbox_bits)[0]
                # print(inp, out)
                mapping = np.concatenate((np.array([0], dtype=np.int32), inp, out))
                key_cnf += self._get_sbox_cnf(0x0, 0x0).translate(mapping)
                array[j] = out.copy()

            for j in range(8, 32):
                array[j] = keyWords[j].copy()
            
            for j in range(4):
                row[j] = get_col(array, j)
                rk_row[j] = get_col(self.r_key[i], j)
            
            rk_row[0] = add_round_constants(rk_row[0], i)
            
            row0r8 = np.roll(row[0], 8)
            row2r16 = np.roll(row[2], 16)
            key_cnf += XorCNF.create_xor(rk_row[0].flatten(), row0r8.flatten(), row[1].flatten())
            key_cnf += XorCNF.create_xor(rk_row[1].flatten(), row[2].flatten())
            key_cnf += XorCNF.create_xor(rk_row[2].flatten(), row2r16.flatten(), row[3].flatten())
            key_cnf += XorCNF.create_xor(rk_row[3].flatten(), row[0].flatten())

            keyWords = self.r_key[i].copy()
            RK.append(keyWords[:16, :])

        self.cnf += key_cnf
        self._round_keys = np.array(RK)

    def _addKey(self, Y, X, K, rc: int) -> None:
        """
        Y = addKey(X, K)
        """
        key_xor_cnf = XorCNF()
        X_flat = X.flatten()
        K_flat = K.flatten()
        Y_flat = Y.flatten()
        key_xor_cnf += XorCNF.create_xor(X_flat, Y_flat, K_flat)
        self.cnf += key_xor_cnf

    def _model_linear_layer(self) -> None:
        for r in range(self.num_rounds):
            permOut = self.applyPerm(self.sbox_out[r])
            self._addKey(permOut, self.sbox_in[r+1], self._round_keys[r], RC[r])

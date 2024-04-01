#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF
from .util import DDT, RC, perm_nibble_16, perm_nibble_16_inv
from ..cipher_model import SboxCipher, DifferentialCharacteristic
log = logging.getLogger(__name__)
class WARP128(SboxCipher):
    cipher_name = "WARP128"
    sbox = np.array([int(x, 16) for x in "cad3ebf789150246"], dtype=np.uint8)
    ddt  = DDT
    block_size = 128
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
        # for i in range(1, self.num_rounds):
        #     lin_input = self.char.sbox_out[i - 1]
        #     lin_output = self.char.sbox_in[i]
        #     print(f'{lin_output = }')
        #     temp = do_linear_layer(lin_input)
        #     print(f'{temp = }')
        #     if not np.all(temp == lin_output):
        #         raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')
        self._create_vars()
        self._model_sboxes()
        self._model_linear_layer()
    def _create_vars(self):
        self.add_index_array('key', (2, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_in', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('X', (1, self.sbox_count, self.sbox_bits))
        self.add_index_array('tweak', (0,))
        # self.pt = np.empty((2*self.sbox_count, 4))
        # for i in range(self.sbox_count):
        #     self.pt[2*i] = self.sbox_in[0][i]
        #     self.pt[2*i+1] = self.X[0][i]
        # self._fieldnames.add('pt')
        # print(self.sbox_in[0])
        # print(self.X)
        # print(self.pt)
    def linear_layer(self, sbox_out, key, xor_in, xor_out, RC0, RC1):
        temp = xor_out.copy()
        xor_out_flat = temp.reshape(-1) # don't use .flatten() here because it creates a copy
        # flip bits according to round constant
        xor_out_flat[0] *= (-1)**(RC0 & 0x1)
        xor_out_flat[1] *= (-1)**((RC0 >> 1) & 0x1)
        xor_out_flat[2] *= (-1)**((RC0 >> 2) & 0x1)
        xor_out_flat[3] *= (-1)**((RC0 >> 3) & 0x1)
        xor_out_flat[4] *= (-1)**(RC1 & 0x1)
        xor_out_flat[5] *= (-1)**((RC1 >> 1) & 0x1)
        xor_out_flat[6] *= (-1)**((RC1 >> 2) & 0x1)
        xor_out_flat[7] *= (-1)**((RC1 >> 3) & 0x1)
        lin_cnf = XorCNF()
        lin_cnf += XorCNF.create_xor(xor_out_flat, xor_in.flatten(), sbox_out.flatten(), key.flatten())
        self.cnf += lin_cnf
    def _model_linear_layer(self):
        for r in range(self.num_rounds):
            if r == 0:
                xor_in = self.X[0].copy()
            else:
                xor_in = perm_nibble_16(self.sbox_in[r-1])
            xor_out = perm_nibble_16_inv(self.sbox_in[r])
            self.linear_layer(self.sbox_out[r], self.key[r%2], xor_in, xor_out, RC[0][r], RC[1][r])
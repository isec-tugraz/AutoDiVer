#!/usr/bin/env python3.
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF
from .util import DDT, RC, perm_nibble_16, perm_nibble_16_inv, perm_nibble_inv
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
        self.add_index_array('key', (2*self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_in', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('X', (self.sbox_count, self.sbox_bits))
        self.add_index_array('Y', (self.sbox_count, self.sbox_bits))
        self.add_index_array('tweak', (0,))
        self.pt = self.get_round_var(self.sbox_in[0], self.X)
        self._fieldnames.add('pt')
        #self.rounds_in = np.empty((self.num_rounds, 2*self.sbox_count, self.sbox_bits))
        #self.rounds_in[0] = self.get_round_var(self.sbox_in[0], self.X[0])
        #for i in range(1, self.num_rounds):
        #    self.rounds_in[i] = self.get_round_var(self.sbox_in[i], perm_nibble_16(self.sbox_in[i-1]))
        #self.rounds_out = np.empty((self.num_rounds, 2*self.sbox_count, self.sbox_bits))
        #for i in range(0, self.num_rounds-1):
        #    self.rounds_out[i] = perm_nibble_inv(self.rounds_in[i])
        ##no permutation at the end
        #self.rounds_out[self.num_rounds-1] = self.get_round_var(self.sbox_in[self.num_rounds-1], self.Y[0])
        # np.empty((2*self.sbox_count, 4), dtype=np.uint32)
        # for i in range(self.sbox_count):
        #     self.pt[2*i] = self.sbox_in[0][i]
        #     self.pt[2*i+1] = self.X[0][i]
        # print(self.sbox_in[0])
        # print(self.X)
        # print(self.pt)
    def get_round_var(self, a, b):
        v = np.empty((2*self.sbox_count, 4), dtype=np.uint32)
        for i in range(self.sbox_count):
            v[2*i] = a[i]
            v[2*i+1] = b[i]
        return v
    def linear_layer(self, sbox_out, key, xor_in, xor_out, RC0, RC1):
        temp = xor_in.copy()
        xor_in_flat = temp.reshape(-1) # don't use .flatten() here because it creates a copy
        # flip bits according to round constant
        xor_in_flat[0] *= (-1)**(RC0 & 0x1)
        xor_in_flat[1] *= (-1)**((RC0 >> 1) & 0x1)
        xor_in_flat[2] *= (-1)**((RC0 >> 2) & 0x1)
        xor_in_flat[3] *= (-1)**((RC0 >> 3) & 0x1)
        xor_in_flat[4] *= (-1)**(RC1 & 0x1)
        xor_in_flat[5] *= (-1)**((RC1 >> 1) & 0x1)
        xor_in_flat[6] *= (-1)**((RC1 >> 2) & 0x1)
        xor_in_flat[7] *= (-1)**((RC1 >> 3) & 0x1)
        lin_cnf = XorCNF()
        lin_cnf += XorCNF.create_xor(xor_in_flat, xor_out.flatten(), sbox_out.flatten(), key.flatten())
        self.cnf += lin_cnf
    def _model_linear_layer(self):
        key = self.key.copy().reshape(2, 16, 4)
        for r in range(self.num_rounds):
            if r == 0:
                xor_in = self.X.copy()
            else:
                xor_in = perm_nibble_16(self.sbox_in[r-1])
            if r == (self.num_rounds - 1):
                xor_out = self.Y.copy()
            else:
                xor_out = perm_nibble_16_inv(self.sbox_in[r+1])
            # print(xor_in)
            # print(xor_out)
            self.linear_layer(self.sbox_out[r], key[r%2], xor_in, xor_out, RC[0][r], RC[1][r])
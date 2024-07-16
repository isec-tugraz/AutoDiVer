#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations

import logging

import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF

from .gift_util import bit_perm, P128, DDT as GIFT_DDT, GIFT_RC
from ..cipher_model import SboxCipher, DifferentialCharacteristic

log = logging.getLogger(__name__)


class Gift128(SboxCipher):
    cipher_name = "GIFT128"
    sbox = np.array([int(x, 16) for x in "1a4c6f392db7508e"], dtype=np.uint8)
    ddt  = GIFT_DDT
    block_size = 128
    key_size = 128

    sbox_bits = 4
    sbox_count = 32

    key: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape

        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')

        for i in range(1, self.num_rounds):
            lin_input = self.char.sbox_out[i - 1].copy()
            lin_output = self.char.sbox_in[i].copy()
            permuted = bit_perm(lin_input)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')


        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('key', (self.sbox_count, self.sbox_bits))

        # print(self.sbox_in)
        # print(self.sbox_out)
        # print(self.key)

        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

        self._model_sboxes()
        self._model_key_schedule()
        self._model_linear_layer()

        self.cnf.nvars = self.numvars

    def applyPerm(self, array: np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.int32]]:
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[P128]
        arrayOut = arrayPermuted.reshape(32, 4)
        return arrayOut

    def _model_key_schedule(self) -> None:
       keyWords = self.key.copy().reshape(8, 16)
       RK = []
       for _ in range(self.num_rounds):
           keyWords32 = keyWords.copy().reshape(4, 32)
           rk = np.empty(len(keyWords32[0]) + len(keyWords32[2]), dtype=keyWords.dtype)
           rk[0::2] = keyWords32[0]
           rk[1::2] = keyWords32[2]
           # print(keyWords32[0])
           # print(keyWords32[2])
           # print(rk)
           # print(rk.shape)

           keyWords[0] = np.roll(keyWords[0], -12)
           keyWords[1] = np.roll(keyWords[1], -2)

           #rotatate the words by 2
           keyWords = np.roll(keyWords, -2, axis=0)
           rk = rk.reshape(32, 2)
           RK.append(rk)

       self._round_keys = np.array(RK)

    def _addKey(self, Y, X, K, RC: int) -> None:
        """
        Y = addKey(X, K)
        """
        X = X.copy()
        X_flat = X.reshape(-1) # don't use .flatten() here because it creates a copy

        # flip bits according to round constant
        X_flat[3]  *= (-1)**(RC & 0x1)
        X_flat[7]  *= (-1)**((RC >> 1) & 0x1)
        X_flat[11] *= (-1)**((RC >> 2) & 0x1)
        X_flat[15] *= (-1)**((RC >> 3) & 0x1)
        X_flat[19] *= (-1)**((RC >> 4) & 0x1)
        X_flat[23] *= (-1)**((RC >> 5) & 0x1)
        X_flat[127] *= (-1)

        key_xor_cnf = XorCNF()
        key_xor_cnf += XorCNF.create_xor(X[:, :1].flatten(), Y[:, :1].flatten())
        key_xor_cnf += XorCNF.create_xor(X[:, 3:].flatten(), Y[:, 3:].flatten())
        key_xor_cnf += XorCNF.create_xor(X[:, 1:3].flatten(), Y[:, 1:3].flatten(), K.flatten())
        self.cnf += key_xor_cnf

    def _model_linear_layer(self) -> None:
        for r in range(self.num_rounds):
            permOut = self.applyPerm(self.sbox_out[r])
            self._addKey(permOut, self.sbox_in[r+1], self._round_keys[r], GIFT_RC[r])

#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations

import logging

import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF

from .gift_util import P64, P128, DDT, GIFT_RC
from .gift_characteristic import Gift64Characteristic, Gift128Characteristic
from ..cipher_model import SboxCipher, DifferentialCharacteristic

log = logging.getLogger(__name__)

class _Gift(SboxCipher):
    sbox = np.array([int(x, 16) for x in "1a4c6f392db7508e"], dtype=np.uint8)
    ddt  = DDT
    key_size = 128
    sbox_bits = 4
    permutation: np.ndarray

    sbox_count: int
    characteristic_type: type

    _round_keys: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        if not isinstance(char, self.characteristic_type):
            raise ValueError(f'expected {self.characteristic_type}, got {type(char)}')

        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds

        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('key', (32, self.sbox_bits))

        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

        self._model_sboxes()
        self._model_key_schedule()
        self._model_linear_layer()

        self.cnf.nvars = self.numvars

    def applyPerm(self, array: np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.int32]]:
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[self.permutation]
        arrayOut = arrayPermuted.reshape(self.sbox_count, 4)
        return arrayOut

    def _model_key_schedule(self) -> None:
        raise NotImplementedError("implement in subclass")

    def _addKey(self, Y, X, K, RC: int) -> None:
        raise NotImplementedError("implement in subclass")

    def _model_linear_layer(self) -> None:
        for r in range(self.num_rounds):
            permOut = self.applyPerm(self.sbox_out[r])
            self._addKey(permOut, self.sbox_in[r+1], self._round_keys[r], GIFT_RC[r])

class Gift128(_Gift):
    cipher_name = "GIFT128"
    characteristic_type = Gift128Characteristic
    block_size = 128
    sbox_count = 32
    permutation = P128

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



class Gift64(_Gift):
    cipher_name = "GIFT64"
    characteristic_type = Gift64Characteristic
    block_size = 64
    sbox_count = 16
    permutation = P64

    def _model_key_schedule(self) -> None:
        keyWords = self.key.copy().reshape(8, 16)

        RK = []
        for _ in range(self.num_rounds):
            rk = np.empty(len(keyWords[0]) + len(keyWords[1]), dtype=keyWords.dtype)
            rk[0::2] = keyWords[0]
            rk[1::2] = keyWords[1]

            keyWords[0] = np.roll(keyWords[0], -12)
            keyWords[1] = np.roll(keyWords[1], -2)

            #rotatate the words by 2
            keyWords = np.roll(keyWords, -2, axis=0)
            rk = rk.reshape(16, 2)
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
        X_flat[63] *= (-1)

        key_xor_cnf = XorCNF()
        key_xor_cnf += XorCNF.create_xor(X[:, 2:].flatten(), Y[:, 2:].flatten())
        key_xor_cnf += XorCNF.create_xor(X[:, :2].flatten(), Y[:, :2].flatten(), K.flatten())
        self.cnf += key_xor_cnf


class Gift128FullKey(_Gift):
    """
    Variant of GIFT-128 where a full key addition with indepnend round keys happens each round.
    Round constants are not modeled.
    """
    cipher_name = "GIFT128-full-key"
    characteristic_type = Gift128Characteristic
    block_size = 128
    sbox_count = 32
    permutation = P128

    round_keys: np.ndarray[Any, np.dtype[np.int32]]

    def _model_key_schedule(self) -> None:
        self.add_index_array('round_keys', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.key = self.round_keys

        self._round_keys = self.round_keys

    def _addKey(self, Y, X, K, RC: int) -> None:
        self.cnf += XorCNF.create_xor(X.flatten(), Y.flatten(), K.flatten())

class Gift64FullKey(_Gift):
    """
    Variant of GIFT-64 where a full key addition with indepnend round keys happens each round.
    Round constants are not modeled.
    """
    cipher_name = "GIFT64-full-key"
    characteristic_type = Gift64Characteristic
    block_size = 64
    sbox_count = 16
    permutation = P64

    round_keys: np.ndarray[Any, np.dtype[np.int32]]

    def _model_key_schedule(self) -> None:
        self.add_index_array('round_keys', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.key = self.round_keys

        self._round_keys = self.round_keys

    def _addKey(self, Y, X, K, RC: int) -> None:
        self.cnf += XorCNF.create_xor(X.flatten(), Y.flatten(), K.flatten())

#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sat_toolkit.formula import XorCNF

from .util import DDT, RC, do_shift_rows, mixing_mat, do_mix_columns
from .generate_perm import permutation
from ..cipher_model import SboxCipher, DifferentialCharacteristic

log = logging.getLogger(__name__)

class Midori128Characteristic(DifferentialCharacteristic):
    def __init__(self, sbox_in: np.ndarray, sbox_out: np.ndarray, file_path: Path|None):
        assert sbox_in.dtype == sbox_out.dtype == np.uint8
        assert sbox_in.shape == sbox_out.shape
        assert sbox_in.shape[-2:] == sbox_out.shape[-2:] == (4, 4)
        assert np.all((sbox_in == 0) == (sbox_out == 0))

        super().__init__(sbox_in, sbox_out, file_path=file_path)

        # verify linear layer
        for i in range(0, self.num_rounds - 1):
            if not np.all(do_mix_columns(do_shift_rows(self.sbox_out[i])) == self.sbox_in[i + 1]):
                raise ValueError(f'linear layer condition violated at sbox_out[{i}] -> sbox_in[{i + 1}]')

        self.original_sbox_in = sbox_in
        self.original_sbox_out = sbox_out

        # transform byte differences into nibble differences
        # (also permute bits according to S-box specification)
        sbox_in = np.array(sbox_in, dtype=np.uint8).swapaxes(-1, -2).reshape(-1, 16)
        sbox_out = np.array(sbox_out, dtype=np.uint8).swapaxes(-1, -2).reshape(-1, 16)

        sbox_in = np.unpackbits(sbox_in, axis=-1, bitorder='little')
        sbox_out = np.unpackbits(sbox_out, axis=-1, bitorder='little')

        perm = permutation()
        sbox_in = sbox_in[:, perm].reshape(-1, 32, 4)
        sbox_out = sbox_out[:, perm].reshape(-1, 32, 4)

        sbox_in = np.packbits(sbox_in, axis=-1, bitorder='little')
        sbox_out = np.packbits(sbox_out, axis=-1, bitorder='little')

        assert sbox_in.shape[-1] == sbox_out.shape[-1] == 1
        sbox_in = sbox_in[..., 0]
        sbox_out = sbox_out[..., 0]

        self.sbox_in = sbox_in
        self.sbox_out = sbox_out

        # np.set_printoptions(formatter={'int': lambda x: f"{x:x}"}, linewidth=300)
        # from IPython import embed; embed(); exit()


    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')

        return cls(sbox_in, sbox_out, file_path=characteristic_path)

class Midori128(SboxCipher):
    cipher_name = "MIDORI128"
    sbox = np.array([int(x, 16) for x in "1053e2f7da9bc846"], dtype=np.uint8)
    ddt  = DDT
    block_size = 128
    key_size = 128

    sbox_bits = 4
    sbox_count = 32

    key: np.ndarray[Any, np.dtype[np.int32]]
    mc_out: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: Midori128Characteristic, **kwargs):
        if not isinstance(char, Midori128Characteristic):
            raise ValueError(f'char must be an instance of Midori128Characteristic, not {type(char)}')

        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape

        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')

        self._create_vars()
        self._key_schedule()

        self._model_add_key()
        self._model_linear_layer()
        self._model_sboxes()

    def _create_vars(self):
        self.add_index_array('key', (1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('mc_out', (self.num_rounds, self.sbox_count, self.sbox_bits))

        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

    def _key_schedule(self) -> None:
        RK = []
        for i in range(self.num_rounds):
            rk = self.key.copy()
            RK.append(rk)
        self._round_keys = np.array(RK)

    def _model_sboxes(self):
        perm = permutation()
        sbox_in_permuted = np.zeros_like(self.sbox_in)
        sbox_out_permuted = np.zeros_like(self.sbox_out)

        for i in range(self.num_rounds):
            sin_flat = self.sbox_in[i].flatten()[perm]
            sout_flat = self.sbox_out[i].flatten()[perm]

            sbox_in_permuted[i] = sin_flat.reshape(32, 4)
            sbox_out_permuted[i] = sout_flat.reshape(32, 4)

        super()._model_sboxes(sbox_in_permuted, sbox_out_permuted)

    def _addKey(self, Y, X, K, RC: np.ndarray):
        X_flat = X.copy().reshape(16, 8)
        for i in range(16):
            X_flat[i][4]  *= np.int8(-1)**(RC[i] & 0x1)
        X_flat = X_flat.flatten()
        key_xor_cnf = XorCNF.create_xor(X_flat, Y.flatten(), K.flatten())
        return key_xor_cnf

    def _model_add_key(self):
        for r in range(self.num_rounds):
            self.cnf += self._addKey(self.mc_out[r], self.sbox_in[r+1], self._round_keys[r], RC[r])

    @staticmethod
    def model_mix_cols(A, B):
        mc_cnf = XorCNF()
        for c in range(4):
            colA = A[:, c]
            colB = B[:, c]
            for r in range(4):
                colA_red = colA[mixing_mat[r] != 0, :]
                assert len(colA_red) == 3
                mc_cnf += XorCNF.create_xor(colB[r], *colA_red)
        return mc_cnf

    def _model_linear_layer(self):
        for r in range(self.num_rounds):
            mc_input = do_shift_rows(self.sbox_out[r].reshape(4, 4, 8).swapaxes(0, 1))
            mc_output = self.mc_out[r].reshape(4, 4, 8).swapaxes(0, 1)

            if r < self.num_rounds - 1:
                self.cnf += self.model_mix_cols(mc_input, mc_output)
            else:
                # no mix columns in the last round
                self.cnf += XorCNF.create_xor(mc_input.flatten(), mc_output.flatten())

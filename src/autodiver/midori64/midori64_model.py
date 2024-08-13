#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations

TYPE_CHECKING=False
if TYPE_CHECKING:
    from typing import Any, Self

import logging
from pathlib import Path

import numpy as np
from sat_toolkit.formula import XorCNF

from .util import DDT, RC, do_shift_rows, mixing_mat, do_linear_layer
from ..cipher_model import SboxCipher, DifferentialCharacteristic

log = logging.getLogger(__name__)

def midori64_mc(inp: int) -> int:
    nibbles = np.array([inp >> (4*i) & 0xf for i in range(16)], dtype=np.uint8)
    nibbles = nibbles[::-1]
    nibbles = nibbles.reshape(4, 4).T

    out = np.zeros_like(nibbles)
    out[0, :] = nibbles[1, :] ^ nibbles[2, :] ^ nibbles[3, :]
    out[1, :] = nibbles[0, :] ^ nibbles[2, :] ^ nibbles[3, :]
    out[2, :] = nibbles[0, :] ^ nibbles[1, :] ^ nibbles[3, :]
    out[3, :] = nibbles[0, :] ^ nibbles[1, :] ^ nibbles[2, :]

    out = out.T.ravel()
    out = out[::-1]
    return sum(int(out[i]) << (4*i) for i in range(16))

def midori64_sr(inp: int) -> int:
    nibbles = np.array([inp >> (4*i) & 0xf for i in range(16)], dtype=np.uint8)
    nibbles = nibbles[::-1]

    perm = np.array([0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8])
    out = nibbles[perm]

    out = out[::-1]
    return sum(int(out[i]) << (4*i) for i in range(16))


def matrix_as_uint64(matrix: np.ndarray) -> int:
    assert np.all(matrix >= 0) and np.all(matrix < 16)
    assert matrix.shape == (4, 4)
    flat = matrix.T.ravel()

    return int(''.join(f'{x:01x}' for x in flat), 16)

class Midori64Characteristic(DifferentialCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint8]] = DDT

    @classmethod
    def load(cls, characteristic_path: Path) -> Self:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']

        sbox_in = np.array(sbox_in, dtype=np.uint8)
        sbox_out = np.array(sbox_out, dtype=np.uint8)

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')

        return cls(sbox_in, sbox_out, file_path=characteristic_path)


class _Midori64Base(SboxCipher):
    cipher_name = "MIDORI64"
    sbox = np.array([int(x, 16) for x in "cad3ebf789150246"], dtype=np.uint8)
    ddt  = DDT
    block_size = 64
    key_size = 128

    sbox_bits = 4

    key: np.ndarray[Any, np.dtype[np.int32]]
    mc_out: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        if not isinstance(char, Midori64Characteristic):
            raise ValueError('char must be of type Midori64Characteristic')
        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape

        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')

        for i in range(1, self.num_rounds):
            lin_input = matrix_as_uint64(self.char.sbox_out[i - 1])
            lin_output = matrix_as_uint64(self.char.sbox_in[i])


            temp = midori64_mc(midori64_sr(lin_input))
            if not temp == lin_output:
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')

        self._create_vars()

        self._model_sboxes()
        self._model_linear_layer()
        self._model_add_key()

    def _create_vars(self):
        self.add_index_array('sbox_in', (self.num_rounds+1, 4, 4, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, 4, 4, self.sbox_bits))
        self.add_index_array('mc_out', (self.num_rounds, 4, 4, self.sbox_bits))

        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

    def _model_add_key(self):
        raise NotImplementedError("Subclasses must implement _model_add_key")

    @staticmethod
    def model_mix_cols(A, B):
        mc_cnf = XorCNF()
        for c in range(4):
            colA = A[:, c]
            colB = B[:, c]
            for r in range(4):
                colA_red = colA[mixing_mat[r] != 0, :]
                assert len(colA_red) == 3
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

            if r < self.num_rounds - 1:
                self.cnf += self.model_mix_cols(mc_input, mc_output)
            else:
                # no mix columns in the last round
                self.cnf += XorCNF.create_xor(mc_input.flatten(), mc_output.flatten())


class Midori64(_Midori64Base):
    def _model_add_key(self):
        # we omit the round key addition at the first and last round
        # as the do not influence the number of solutions and the corresponding
        # values can always be calculated in a post-processing step
        self.add_index_array('key', (2, 4, 4, self.sbox_bits)) # different amount of model vars depending on version

        for r in range(self.num_rounds):
            inp = self.mc_out[r].swapaxes(0, 1).flatten()
            out = self.sbox_in[r + 1].swapaxes(0, 1).flatten()
            key = self.key[r % 2].swapaxes(0, 1).flatten()
            rc = RC[r].swapaxes(0, 1).flatten()

            for i in range(16):
                inp[4*i]  *= np.int8(-1)**(rc[i] & 0x1)

            if r < self.num_rounds - 1:
                self.cnf += XorCNF.create_xor(inp, out, key)
            else:
                # all equal
                self.cnf += XorCNF.create_xor(inp, out)


class Midori64LongKey(_Midori64Base):
    round_key: np.ndarray[Any, np.dtype[np.int32]]

    def _model_add_key(self):
        # we omit the round key addition at the first and last round
        # as the do not influence the number of solutions and the corresponding
        # values can always be calculated in a post-processing step
        self.add_index_array('round_key', (self.num_rounds - 1, 4, 4, self.sbox_bits)) # different amount of model vars depending on version
        self.key = self.round_key
        self._fieldnames.add('key')

        for r in range(self.num_rounds):
            inp = self.mc_out[r].swapaxes(0, 1).flatten()
            out = self.sbox_in[r + 1].swapaxes(0, 1).flatten()

            if r == self.num_rounds - 1:
                # all equal
                self.cnf += XorCNF.create_xor(inp, out)
                return

            key = self.key[r].swapaxes(0, 1).flatten()
            # rc = RC[r].swapaxes(0, 1).flatten()

            # for i in range(16):
            #     inp[4*i]  *= np.int8(-1)**(rc[i] & 0x1)

            self.cnf += XorCNF.create_xor(inp, out, key)

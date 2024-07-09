#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import logging
from typing import Any
from pathlib import Path

from .present_cipher import s_box, p_layer_order
from .present_util import PERM, INV_PERM, bit_perm
from ..cipher_model import SboxCipher, DifferentialCharacteristic

import numpy as np
from sat_toolkit.formula import XorCNF


log = logging.getLogger(__name__)
PRESENT_DDT = np.array(
    [[16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [ 0,  0,  0,  4,  0,  0,  0,  4,  0,  4,  0,  0,  0,  4,  0,  0],
     [ 0,  0,  0,  2,  0,  4,  2,  0,  0,  0,  2,  0,  2,  2,  2,  0],
     [ 0,  2,  0,  2,  2,  0,  4,  2,  0,  0,  2,  2,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0,  4,  2,  2,  0,  2,  2,  0,  2,  0,  2,  0],
     [ 0,  2,  0,  0,  2,  0,  0,  0,  0,  2,  2,  2,  4,  2,  0,  0],
     [ 0,  0,  2,  0,  0,  0,  2,  0,  2,  0,  0,  4,  2,  0,  0,  4],
     [ 0,  4,  2,  0,  0,  0,  2,  0,  2,  0,  0,  0,  2,  0,  0,  4],
     [ 0,  0,  0,  2,  0,  0,  0,  2,  0,  2,  0,  4,  0,  2,  0,  4],
     [ 0,  0,  2,  0,  4,  0,  2,  0,  2,  0,  0,  0,  2,  0,  4,  0],
     [ 0,  0,  2,  2,  0,  4,  0,  0,  2,  0,  2,  0,  0,  2,  2,  0],
     [ 0,  2,  0,  0,  2,  0,  0,  0,  4,  2,  2,  2,  0,  2,  0,  0],
     [ 0,  0,  2,  0,  0,  4,  0,  2,  2,  2,  2,  0,  0,  0,  2,  0],
     [ 0,  2,  4,  2,  2,  0,  0,  2,  0,  0,  2,  2,  0,  0,  0,  0],
     [ 0,  0,  2,  2,  0,  0,  2,  2,  2,  2,  0,  0,  2,  2,  0,  0],
     [ 0,  4,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  4]])


class PresentCharacteristic(DifferentialCharacteristic):
    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']
        sbox_in = np.array(sbox_in, dtype=np.uint8)
        sbox_out = np.array(sbox_out, dtype=np.uint8)

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')
        if len(sbox_in.shape) != 2 or len(sbox_out.shape) != 2:
            raise ValueError('sbox_in and sbox_out must have 2 dimensions')
        if sbox_in.shape[1] != 16 or sbox_out.shape[1] != 16:
            raise ValueError('sbox_in and sbox_out must have 16 s-boxes')

        num_rounds = sbox_in.shape[0]
        permuted = bit_perm(sbox_out)

        for i in range(1, num_rounds):
            lin_input = sbox_out[i - 1]
            lin_output = sbox_in[i]
            permuted = bit_perm(lin_input)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')


        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')
        return cls(sbox_in, sbox_out, file_path=characteristic_path)


class Present(SboxCipher):
    sbox = np.array([int(x, 16) for x in "c56b90ad3ef84712"], dtype=np.uint8)
    ddt  = PRESENT_DDT
    block_size = 64
    sbox_bits = 4
    sbox_count = 16

    long_round_keys: np.ndarray[Any, np.dtype[np.int32]]
    round_keys: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        if not isinstance(char, PresentCharacteristic):
            raise ValueError('char must be a PresentCharacteristic')

        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds

        assert self.char.sbox_in.shape == self.char.sbox_out.shape

        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')

        # for i in range(1, self.num_rounds):
        #     lin_input = self.char.sbox_out[i - 1]
        #     lin_output = self.char.sbox_in[i]
        #     permuted = PERM[lin_input]
        #     if not np.all(permuted == lin_output):
        #         raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')

        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds + 1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))

        self.add_index_array('long_round_keys', (self.num_rounds + 1, self.key_size))
        self._fieldnames.add('round_keys')
        self._fieldnames.add('key')
        self.round_keys = self.long_round_keys[:, self.key_size-self.block_size:]
        self.key = self.long_round_keys[0]

        self.add_index_array('tweak', (0,))

        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

        self._model_sboxes()
        if self.key_size == 80:
            self._model_key_schedule80()
        elif self.key_size == 128:
            self._model_key_schedule128()
        else:
            raise ValueError('key_size must be 80 or 128')
        self._model_linear_layer()

    @classmethod
    def _fmt_arr(cls, arr: np.ndarray, cellsize: int):
        if cellsize == 0 and len(arr) == 0:
            return ''

        if cellsize == 4:
            int_val = sum(int(x) << i * 4 for i, x in enumerate(arr))
            return f'0x{int_val:016x}'

        assert np.all(arr < 256) and np.all(arr >= 0)
        int_val = sum(int(x) << i * 8 for i, x in enumerate(arr))
        if cellsize == 80:
            return f'0x{int_val:020x}'
        if cellsize == 128:
            return f'0x{int_val:032x}'

        return f'0x{int_val:0{len(arr // 4)}x}'

        raise ValueError(f'cellsize must be 4 or 8 not {cellsize}')

    def applyPerm(self, array: np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.int32]]:
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[PERM]
        arrayOut = arrayPermuted.reshape(16, 4)
        return arrayOut

    def _model_key_schedule80(self) -> None:
        key_schedule_cnf = XorCNF()
        for rnd in range(self.num_rounds):
            in_key = self.long_round_keys[rnd]
            rotated_key = np.roll(in_key, 61)
            rc = (rnd + 1) << 15
            rc_arr = np.array([(rc >> i) & 1 for i in range(80)], dtype=np.int8)
            sb_inp_key = rotated_key * (-1)**rc_arr
            sb_out_key = self.long_round_keys[rnd + 1]

            # the lower bits are equal
            key_schedule_cnf += XorCNF.create_xor(sb_inp_key[:76], sb_out_key[:76])
            # s-box for the 4 most significant bits
            mapping = np.concatenate((np.array([0], dtype=np.int32), sb_inp_key[76:], sb_out_key[76:]))
            key_schedule_cnf += self._get_sbox_cnf(0, 0).translate(mapping)
        self.cnf += key_schedule_cnf


    def _model_key_schedule128(self) -> None:
        raise NotImplementedError

    def _addKey(self, inp, out, key) -> None:
        key_xor_cnf = XorCNF()
        key_xor_cnf += XorCNF.create_xor(inp.flatten(), out.flatten(), key)
        self.cnf += key_xor_cnf

    def _model_linear_layer(self) -> None:
        for r in range(self.num_rounds):
            permOut = self.applyPerm(self.sbox_out[r])
            self._addKey(permOut, self.sbox_in[r+1], self.round_keys[r + 1])

class Present80(Present):
    cipher_name = 'PRESENT-80'
    key_size = 80

class Present128(Present):
    cipher_name = 'PRESENT-128'
    key_size = 128

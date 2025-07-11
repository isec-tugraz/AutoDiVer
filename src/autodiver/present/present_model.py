#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations

TYPE_CHECKING=False
if TYPE_CHECKING:
    from typing import Any

import logging
from pathlib import Path

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
     [ 0,  4,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  4]], dtype=np.uint8)


class PresentCharacteristic(DifferentialCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint8]] = PRESENT_DDT

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

        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds + 1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))

        # key schedule vars are added by subclasses in _model_key_schedule

        self.add_index_array('tweak', (0,))

        self.add_index_array('pt', (self.sbox_count, self.sbox_bits))

        self._model_sboxes()
        self._model_key_schedule()
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

    def _model_key_schedule(self) -> None:
        raise NotImplementedError("Subclasses must implement _model_key_schedule")

    def _model_linear_layer(self) -> None:
        self.cnf += XorCNF.create_xor(self.pt.flatten(), self.sbox_in[0].flatten(), self.round_keys[0])
        for r in range(self.num_rounds):
            permOut = self.applyPerm(self.sbox_out[r])
            self.cnf += XorCNF.create_xor(permOut.flatten(), self.sbox_in[r+1].flatten(), self.round_keys[r + 1].flatten())

class Present80(Present):
    cipher_name = 'PRESENT-80'
    key_size = 80

    _long_round_keys: np.ndarray[Any, np.dtype[np.int32]]

    def _model_key_schedule(self):
        self.add_index_array('_long_round_keys', (self.num_rounds + 1, self.key_size))

        self.key = self._long_round_keys[0]
        self.round_keys = self._long_round_keys[:, self.key_size-self.block_size:]


        self._fieldnames.add('round_keys')
        self._fieldnames.add('key')

        key_schedule_cnf = XorCNF()
        for rnd in range(self.num_rounds):
            in_key = self._long_round_keys[rnd]
            rotated_key = np.roll(in_key, 61)
            rc = (rnd + 1) << 15
            rc_arr = np.array([(rc >> i) & 1 for i in range(80)], dtype=np.int8)
            sb_inp_key = rotated_key * (-1)**rc_arr
            sb_out_key = self._long_round_keys[rnd + 1]

            # the lower bits are equal
            key_schedule_cnf += XorCNF.create_xor(sb_inp_key[:76], sb_out_key[:76])
            # s-box for the 4 most significant bits
            mapping = np.concatenate((np.array([0], dtype=np.int32), sb_inp_key[76:], sb_out_key[76:]))
            key_schedule_cnf += self._get_sbox_cnf(0, 0).translate(mapping)
        self.cnf += key_schedule_cnf

class PresentLongKey(Present):
    cipher_name = 'PRESENT-long-key'

    def _model_key_schedule(self):
        self.key_size = (self.num_rounds + 1) * self.block_size
        self.add_index_array('round_keys', (self.num_rounds + 1, self.block_size))
        assert self.key_size == self.round_keys.size

        self.key = self.round_keys

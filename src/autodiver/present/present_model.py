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

from .present_util import PERM, INV_PERM, bit_perm, PRESENT_DDT
from ..cipher_model import SboxCipher, DifferentialCharacteristic
from .present_characteristic import PresentCharacteristic

import numpy as np
from sat_toolkit.formula import XorCNF


log = logging.getLogger(__name__)

class Present(SboxCipher):
    sbox = np.array([int(x, 16) for x in "c56b90ad3ef84712"], dtype=np.uint8)
    ddt  = PRESENT_DDT
    block_size = 64
    sbox_bits = 4
    sbox_count = 16

    round_keys: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        if not isinstance(char, PresentCharacteristic):
            raise ValueError('char must be a DifferentialCharacteristic or PresentCharacteristic')

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

        if self.search_char:
            self.add_index_array("ddt_weights", (self.num_rounds, self.sbox_count, self.num_bits_ddt_weights))
            self._model_ddt()
            self.add_index_array('key', (0,))
        else:
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
        if self.search_char:
            self.cnf += XorCNF.create_xor(self.pt.flatten(), self.sbox_in[0].flatten())
            for r in range(self.num_rounds):
                permOut = self.applyPerm(self.sbox_out[r])
                self.cnf += XorCNF.create_xor(permOut.flatten(), self.sbox_in[r + 1].flatten())
        else:
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

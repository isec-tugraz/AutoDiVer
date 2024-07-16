#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for Ascon
"""
from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

import numpy as np
from sat_toolkit.formula import XorCNF
import sys

from ..util import get_ddt
from ..cipher_model import SboxCipher, DifferentialCharacteristic

log = logging.getLogger(__name__)

def rotr(val: np.uint64, r: int):
    val_int = int(val)
    res_int = (val_int >> r) | ((val_int & (1<<r)-1) << (64-r))
    return np.uint64(res_int)

def ascon_linear_layer(inp: np.ndarray[Any, np.dtype[np.uint64]]) -> np.ndarray[Any, np.dtype[np.uint64]]:
    S = inp.copy()

    S[0] ^= rotr(S[0], 19) ^ rotr(S[0], 28)
    S[1] ^= rotr(S[1], 61) ^ rotr(S[1], 39)
    S[2] ^= rotr(S[2],  1) ^ rotr(S[2],  6)
    S[3] ^= rotr(S[3], 10) ^ rotr(S[3], 17)
    S[4] ^= rotr(S[4],  7) ^ rotr(S[4], 41)

    return S

class AsconCharacteristic(DifferentialCharacteristic):
    def __init__(self, sbox_in: np.ndarray[Any, np.dtype[np.uint64]], sbox_out: np.ndarray[Any, np.dtype[np.uint64]], **kwargs):
        # do reverse bit slicing
        assert sys.byteorder == 'little'
        assert sbox_in.dtype.byteorder in ['<', '=']
        assert sbox_out.dtype.byteorder in ['<', '=']
        assert sbox_in.dtype == sbox_out.dtype == np.uint64

        for i in range(len(sbox_in) - 1):
            assert np.all(ascon_linear_layer(sbox_out[i]) == sbox_in[i + 1])

        # unpack bits
        sbox_in_bits = np.unpackbits(sbox_in.view(np.uint8), axis=-1, bitorder='little').reshape(-1, 5, 64)
        sbox_out_bits = np.unpackbits(sbox_out.view(np.uint8), axis=-1, bitorder='little').reshape(-1, 5, 64)

        # bit slice
        sbox_in_bits = sbox_in_bits.swapaxes(-1, -2)
        sbox_out_bits = sbox_out_bits.swapaxes(-1, -2)

        # swap order because x0 is MSB and x4 is LSB
        sbox_in_bits = sbox_in_bits[..., ::-1]
        sbox_out_bits = sbox_out_bits[..., ::-1]

        # pack bits
        sbox_in_unbitsliced = np.packbits(sbox_in_bits, axis=-1, bitorder='little')[..., 0]
        sbox_out_unbitsliced = np.packbits(sbox_out_bits, axis=-1, bitorder='little')[..., 0]

        assert np.all(Ascon.ddt[sbox_in_unbitsliced, sbox_out_unbitsliced] > 0)

        super().__init__(sbox_in_unbitsliced, sbox_out_unbitsliced, **kwargs)

    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']

        sbox_in = np.array(sbox_in, dtype=np.uint64)
        sbox_out = np.array(sbox_out, dtype=np.uint64)

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')

        return cls(sbox_in, sbox_out, file_path=characteristic_path)

class Ascon(SboxCipher):
    cipher_name = "Ascon"
    sbox = np.array(bytearray.fromhex("040b1f141a1509021b0508121d03061c1e13070e000d1118100c0119160a0f17"))
    ddt  = get_ddt(sbox)
    block_size = 320
    key_size = 0

    sbox_bits = 5
    sbox_count = 64

    key: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: AsconCharacteristic, **kwargs):
        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape == (self.num_rounds, 64)
        assert self.char.sbox_in.dtype == self.char.sbox_out.dtype == np.uint8

        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds, 5, 64))
        self.add_index_array('sbox_out', (self.num_rounds, 5, 64))

        self.add_index_array('key', (0,))
        self.add_index_array('tweak', (0,))


        # bitsliced sbox input with round constant


        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

        self._model_sboxes()
        self._model_linear_layer()

    @classmethod
    def _fmt_arr(cls, arr: np.ndarray, cellsize: int):
        if cellsize == 0 and len(arr) == 0:
            return ''

        if cellsize == 4:
            return ''.join(f'{x:01x}' for x in arr.flatten())
        if cellsize == 8:
            return ''.join(f'{x:02x}' for x in arr.flatten())


        if arr.dtype == np.uint64 and arr.ndim == 1:
            return ' '.join(f'{x:016x}' for x in arr)

        raise ValueError(f'cellsize must be 4 or 8 not {cellsize} -- (shape: {arr.shape})')


    def _model_sboxes(self, sbox_in: None|np.ndarray=None, sbox_out: None|np.ndarray=None) -> None:
        sbox_in = sbox_in.copy() if sbox_in is not None else self.sbox_in.copy()
        sbox_out = sbox_out.copy() if sbox_out is not None else self.sbox_out.copy()

        # round constants
        constants = np.array([0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b], dtype=np.uint8)
        constants = np.unpackbits(constants.reshape(-1, 1), axis=-1, bitorder='little')
        constants = np.pad(constants, ((0, 0), (0, 64 - 8))) # type:ignore

        for r in range(self.num_rounds):
            sbox_in[r, 2] *= np.int8(-1)**constants[12 - self.num_rounds + r]

        # swap axes for bitsliced sboxes
        # swap bits for compatibility with big-endian s-box table
        self.sbox_in_bitsliced = sbox_in.swapaxes(-1, -2)[..., ::-1]
        self.sbox_out_bitsliced = sbox_out.swapaxes(-1, -2)[..., ::-1]

        self._fieldnames.add('sbox_in_bitsliced')
        self._fieldnames.add('sbox_out_bitsliced')

        super()._model_sboxes(self.sbox_in_bitsliced, self.sbox_out_bitsliced)


    def _model_linear_layer(self) -> None:
        cnf = XorCNF()
        for r in range(self.num_rounds - 1):
            lin_in = self.sbox_out[r]
            lin_out = self.sbox_in[r+1]

            shift_1 = (19, 61, 1, 10, 7)
            shift_2 = (28, 39, 6, 17, 41)

            for (inp, out, a, b) in zip(lin_in, lin_out, shift_1, shift_2):
                inp2 = np.roll(inp, 64 - a)
                inp3 = np.roll(inp, 64 - b)
                cnf += XorCNF.create_xor(inp, inp2, inp3, out)

        self.cnf += cnf

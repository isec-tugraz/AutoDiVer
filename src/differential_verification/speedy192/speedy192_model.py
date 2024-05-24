#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF
from .ddt import DDT
from .util import SBOX, RC, do_shift_cols, update_key
from ..cipher_model import SboxCipher, DifferentialCharacteristic
log = logging.getLogger(__name__)
class Speedy192Characteristic(DifferentialCharacteristic):
    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        trail = []
        with open(characteristic_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                assert len(line) == 64
                line2 = [line[i:i+2] for i in range(0, len(line), 2)]
                line_deltas = [int(l, 16) for l in line2]
                # print(line_deltas)
                trail.append(line_deltas)
        trail = np.array(trail)
        if len(trail) % 2 != 0:
            log.error(f'expected an even number of differences in {characteristic_path!r}')
            raise ValueError(f'expected an even number of differences in {characteristic_path!r}')
        sbox_in = trail[0::2]
        sbox_out = trail[1::2]
        return cls(sbox_in, sbox_out, file_path=characteristic_path)
class Speedy192(SboxCipher):
    cipher_name = "SPEEDY192"
    sbox = SBOX.copy()
    # print(f'{sbox}')
    ddt  = DDT
    block_size = 192
    key_size = 192
    sbox_bits = 6
    sbox_count = 32
    # key: np.ndarray[Any, np.dtype[np.int32]]
    # mc_out: np.ndarray[Any, np.dtype[np.int32]]
    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds # this is actually number of sboxes
        assert char.num_rounds%2 == 0 #nimber of sbox layers should be even
        self.num_rounds_full = char.num_rounds//2
        assert self.char.sbox_in.shape == self.char.sbox_out.shape
        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')
        self._create_vars()
        self._key_schedule()
        self._model_sboxes()
        self._model_sc()  #ShiftColumn
        self._model_add_key()
        self._model_linear_layer()
    def _create_vars(self):
        self.add_index_array('key', (1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_in', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out',(self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('mc_out', (self.num_rounds_full - 1, self.sbox_count, self.sbox_bits))
        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')
    def _key_schedule(self) -> None:
        RK = []
        rk = self.key[0].copy()
        # print(rk)
        for i in range(self.num_rounds_full):
            rk = update_key(rk)
            RK.append(rk)
        self._round_keys = np.array(RK)
    def _model_sc(self):
        for i in range(self.num_rounds_full):
            """
            Note that speedy has two ShiftColumn in each full round. But
            we model only one of then which is between two Sboxes. The other
            ShiftColumn applied during MixColumn operation
            """
            scin_flat = do_shift_cols(self.sbox_out[2*i]).flatten()
            scout_flat = self.sbox_in[2*i+1].flatten()
            sc_cnf = XorCNF.create_xor(scin_flat, scout_flat)
            self.cnf += sc_cnf
    def _addKey(self, Y, X, K, RC: np.ndarray):
        X_flat = X.copy().reshape(32, 6)
        for i in range(32):
            X_flat[i][0]  *= (-1)**(RC[i] & 0x1)
            X_flat[i][1]  *= (-1)**((RC[i]>>1) & 0x1)
            X_flat[i][2]  *= (-1)**((RC[i]>>2) & 0x1)
            X_flat[i][3]  *= (-1)**((RC[i]>>3) & 0x1)
            X_flat[i][4]  *= (-1)**((RC[i]>>4) & 0x1)
            X_flat[i][5]  *= (-1)**((RC[i]>>5) & 0x1)
        X_flat = X_flat.flatten()
        key_xor_cnf = XorCNF.create_xor(X_flat, Y.flatten(), K.flatten())
        return key_xor_cnf
    def _model_add_key(self):
        for r in range(self.num_rounds_full -1):
            #No need to model AddKey for first and last round
            self.cnf += self._addKey(self.mc_out[r], self.sbox_in[2*(r+1)], self._round_keys[r], RC[r])
    @staticmethod
    def model_mix_cols(A, B):
        alphas = [0, 1, 5, 9, 15, 21, 26]
        mc_cnf = XorCNF()
        for c in range(6):
            colA = A[:, c]
            colB = B[:, c]
            # print(f'{colA}')
            for r in range(32):
                colA_red = np.empty(len(alphas), dtype=np.uint32)
                for i in range(len(alphas)):
                    colA_red[i] = colA[(r + alphas[i])%32]
                # print(f'{colB[r]}', "===>", f'{colA_red}')
                mc_cnf += XorCNF.create_xor([colB[r]], *colA_red)
        return mc_cnf
    def _model_linear_layer(self):
        for r in range(self.num_rounds_full - 1):
            mc_input = do_shift_cols(self.sbox_out[2*r + 1])
            mc_output = self.mc_out[r].copy()
            self.cnf += self.model_mix_cols(mc_input, mc_output)
    @classmethod
    def _fmt_arr(cls, arr: np.ndarray, cellsize: int):
        if cellsize == 0 and len(arr) == 0:
            return ''
        if cellsize == 4:
            return ''.join(f'{x:01x}' for x in arr.flatten())
        if cellsize == 8:
            return ''.join(f'{x:02x}' for x in arr.flatten())
        if cellsize == 6:
            return ''.join(f'{x:02x}' for x in arr.flatten())
        raise ValueError(f'cellsize must be 4, 6, or 8 not {cellsize}')
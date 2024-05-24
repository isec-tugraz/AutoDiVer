#!/usr/bin/env python3
"""
This script tries to find a collision for RomulusHash based on a bitwise dual
characteristic in numpy's .npz file format.
"""
from __future__ import annotations
from binascii import hexlify
from itertools import product
from pathlib import Path
from typing import Any, Literal
import logging
import zipfile
from sat_toolkit.formula import XorCNF, CNF, Truthtable
import numpy as np
import numpy.typing as npt
from ..cipher_model import SboxCipher, DifferentialCharacteristic
from ..util import Model
from .constants import apply_perm, connection_poly_4, connection_poly_8, do_mix_cols, do_shift_rows, expanded_rc, get_ddt, mixing_mat, skinny_verbose, tweakey_mask, tweakey_perm, update_tweakey
from .util import sbox8, sbox4, LfsrState, get_solution_set
log = logging.getLogger(__name__)
contexts = []
interrupted = False
sbox8_dnf = Truthtable.from_indices(16, (sbox8.astype(np.uint16) << 0) | (np.arange(256, dtype=np.uint16) << 8))
sbox8_cnf = sbox8_dnf.to_cnf()
sbox4_dnf = Truthtable.from_indices(8, (sbox4.astype(np.uint16) << 0) | (np.arange(16, dtype=np.uint16) << 4))
sbox4_cnf = sbox4_dnf.to_cnf()
sboxes = {8: sbox8, 4: sbox4}
sbox_cnfs = {8: sbox8_cnf, 4: sbox4_cnf}
class _SkinnyBaseCharacteristic(DifferentialCharacteristic):
    block_size = 0
    tweakeys: np.ndarray
    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        rounds = 100
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']
            tweakeys = f['tweakeys']
        sbox_in = np.array(sbox_in, dtype=np.int8)[:rounds]
        sbox_out = np.array(sbox_out, dtype=np.int8)[:rounds]
        tweakeys = np.array(tweakeys, dtype=np.int8)[:rounds]
        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')
        return cls(sbox_in, sbox_out, tweakeys, file_path=characteristic_path)
    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike, tweakeys: npt.ArrayLike, **kwargs):
        super().__init__(sbox_in, sbox_out, **kwargs)
        self.tweakeys = np.array(tweakeys, dtype=np.uint8)
        if self.tweakeys.shape != (self.num_rounds, 3, 4, 4):
            raise ValueError('tweakeys must have shape (num_rounds, 3, 4, 4)')
        # sanity check characteristic
        for i in range(len(self.tweakeys) - 1):
            assert np.all(self.tweakeys[i + 1] == update_tweakey(self.tweakeys[i], self.block_size)), f'tweakey update check failed at round {i}'
            rtk = np.bitwise_xor.reduce(self.tweakeys[i], axis=0) & tweakey_mask
            assert np.all(self.sbox_in[i + 1] == do_mix_cols(do_shift_rows(self.sbox_out[i] ^ rtk)))
        self.num_rounds = len(self.sbox_in)
class Skinny128Characteristic(_SkinnyBaseCharacteristic):
    block_size = 128
class Skinny64Characteristic(_SkinnyBaseCharacteristic):
    block_size = 64
class SkinnyBase(SboxCipher):
    sbox: np.ndarray
    ddt: np.ndarray
    block_size = 128
    key_size = 128
    tweak_size = 256
    sbox_bits = 8
    connection_poly: np.ndarray
    _tk2: np.ndarray
    _tk3: np.ndarray
    def __init__(self, char: _SkinnyBaseCharacteristic):
        super().__init__(char)
        self.numrounds = len(char.sbox_in)
        self._nldtool_char = None
        self._char = None
        self._sbox_in = char.sbox_in
        self._sbox_out = char.sbox_out
        self._unmasked_tweakey = np.bitwise_xor.reduce(char.tweakeys, axis=1)
        self._tweakey = [np.bitwise_and(tk, tweakey_mask) for tk in self._unmasked_tweakey]
        self.num_rounds = char.num_rounds
        self._total_log_prob = None
        self._create_vars()
        self._model_linear_layer()
        self._model_sboxes()
        # add lfsr model
        lfsr_cnf = CNF()
        for lfsr in self._tk2_lfsrs:
            for constraint in lfsr.get_constraints():
                lfsr_cnf += constraint.to_cnf()
        for lfsr in self._tk3_lfsrs:
            for constraint in lfsr.get_constraints():
                lfsr_cnf += constraint.to_cnf()
        self.cnf += lfsr_cnf
    def _create_vars(self):
        rnds = self.numrounds
        self.add_index_array('sbox_in', (self.num_rounds + 1, 4, 4, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, 4, 4, self.sbox_bits))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')
        self.ct = self.sbox_in[-1]
        self._fieldnames.add('ct')
        lfsr_updates = (self.numrounds - 1) // 2 + 1
        self.add_index_array('key', (4, 4, self.sbox_bits))
        self.add_index_array('_tk2', (4, 4, self.sbox_bits + lfsr_updates))
        self.add_index_array('_tk3', (4, 4, self.sbox_bits + lfsr_updates))
        self.tweak = np.array([self._tk2[..., :self.sbox_bits], self._tk3[..., :self.sbox_bits]])
        self._fieldnames.add('tweak')
        tk2_tmp = self._tk2.reshape(16, self.sbox_bits + lfsr_updates)
        tk3_tmp = self._tk3.reshape(16, self.sbox_bits + lfsr_updates)
        self._tk2_lfsrs = [LfsrState(f'tk2_{i}', self.connection_poly[::-1].tolist(), tk2_tmp[i]) for i in range(16)]
        self._tk3_lfsrs = [LfsrState(f'tk3_{i}', self.connection_poly[::-1].tolist(), tk3_tmp[i]) for i in range(16)]
        self.round_tweakeys = []
        for rnd in range(rnds):
            key = self.key.reshape(16, self.sbox_bits)
            # rows 2, 3 of the base tweakey are updated one round earlier
            # corresponding to indices in range(8, 16)
            # tk2 = [self._tk2_lfsrs[i].get_bit_range((rnd + (i in range(8, 16))) // 2) for i in range(16)]
            tk2 = [self._tk2_lfsrs[i].get_bit_range(len(self._tk2_lfsrs[i].vars) - self.sbox_bits - (rnd + (i in range(8, 16))) // 2, self.sbox_bits) for i in range(16)]
            tk3 = [self._tk3_lfsrs[i].get_bit_range((rnd + (i in range(8, 16))) // 2, self.sbox_bits) for i in range(16)]
            # tk3 = [self._tk3_lfsrs[i].get_bit_range((rnds - 1) // 2 - (rnd + (i in range(8, 16))) // 2) for i in range(16)]
            key = apply_perm(key, tweakey_perm, rnd)
            tk2 = apply_perm(tk2, tweakey_perm, rnd)
            tk3 = apply_perm(tk3, tweakey_perm, rnd)
            self.round_tweakeys.append((key, tk2, tk3))
        self.round_tweakeys = np.array(self.round_tweakeys).reshape(rnds, 3, 4, 4, self.sbox_bits)
        # from IPython import embed; embed(); raise SystemExit()
        self.m1 = self.round_tweakeys[0][1]
        self.m2 = self.round_tweakeys[0][2]
    def get_model(self, raw_model: np.ndarray[Any, np.dtype[np.uint8]], *, bitorder: Literal['big', 'little']='little') -> Model:
        model = super().get_model(raw_model, bitorder=bitorder)
        rtks = np.packbits(model.raw_model[self.round_tweakeys], axis=-1, bitorder='little')[..., 0]
        model.round_tweakeys = rtks # type: ignore
        return model
    def _model_linear_layer(self):
        for rnd in range(self.numrounds):
            rtks = self.round_tweakeys[rnd]
            in_rcs = expanded_rc[rnd]
            sb_mc_input = np.zeros((4, 4), object)
            for row, col in product(range(4), range(4)):
                sb_mc_input[row, col] = [self.sbox_out[rnd, row, col]]
            for row, col in product(range(2), range(4)):
                for i in range(3):
                    sb_mc_input[row, col].append(rtks[i, row, col])
            sb_mc_input = do_shift_rows(sb_mc_input)
            in_rcs = do_shift_rows(in_rcs)
            lin_layer_cnf = XorCNF()
            for col in range(4):
                for row in range(4):
                    mc_out_var = self.sbox_in[rnd + 1][row, col]
                    mc_in_vars = sb_mc_input[mixing_mat[row] != 0, col]
                    mc_in_vars = sum(mc_in_vars, start=[])
                    constant = np.bitwise_xor.reduce(in_rcs[mixing_mat[row] != 0, col])
                    constant = np.unpackbits(constant, bitorder='little')[:self.sbox_bits]
                    lin_layer_cnf += XorCNF.create_xor(mc_out_var, *mc_in_vars, rhs=constant.astype(np.int32))
            self.cnf += lin_layer_cnf
    @staticmethod
    def _get_cnf(delta_in: int, delta_out: int, cellsize) -> CNF:
        if delta_in == delta_out == 0:
            return sbox_cnfs[cellsize]
        sbox = sboxes[cellsize]
        in_vals = get_solution_set(sbox, delta_in, delta_out)
        dnf = Truthtable.from_indices(2 * cellsize, (sbox[in_vals].astype(np.uint16) << 0) | (in_vals.astype(np.uint16) << cellsize))
        cnf = dnf.to_cnf()
        return cnf
    def _model_sboxes(self):
        for rnd in range(self.numrounds):
            for row, col in product(range(4), range(4)):
                sbox_in = self._sbox_in[rnd, row, col]
                sbox_out = self._sbox_out[rnd, row, col]
                cnf = self._get_cnf(sbox_in, sbox_out, self.sbox_bits)
                variables = [0] + self.sbox_out[rnd, row, col].tolist() + self.sbox_in[rnd, row, col].tolist()
                self.cnf += cnf.translate(variables)
    def get_rtk_value(self, m: Model, rnd: int):
        return np.array([rtk.get_value(m) for rtk in self.round_tweakeys[rnd]])
    def _get_random_pt(self):
        pt = np.zeros_like(self.char.sbox_in[0])
        pt = pt.reshape(-1)
        for i, (di, do) in enumerate(zip(self.char.sbox_in[0].flatten(), self.char.sbox_out[0].flatten())):
            x = np.arange(len(self.sbox), dtype=np.uint8)
            x_set, = np.where(self.sbox[x] ^ self.sbox[x ^ di] == do)
            x = np.random.choice(x_set)
            pt[i] = x
        assert np.all(self.sbox[pt] ^ self.sbox[pt ^ self.char.sbox_in[0].ravel()] == self.char.sbox_out[0].ravel())
        return np.unpackbits(pt, bitorder='little')
class Skinny128(SkinnyBase):
    sbox = sbox8
    ddt = get_ddt(sbox8)
    connection_poly = connection_poly_8
    block_size = 128
    key_size = 128
    tweak_size = 256
    sbox_bits = 8
class Skinny64(SkinnyBase):
    sbox = sbox4
    ddt = get_ddt(sbox4)
    connection_poly = connection_poly_4
    block_size = 64
    key_size = 64
    tweak_size = 128
    sbox_bits = 4
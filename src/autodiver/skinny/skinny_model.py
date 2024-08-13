#!/usr/bin/env python3
"""
This script tries to find a collision for RomulusHash based on a bitwise dual
characteristic in numpy's .npz file format.
"""
from __future__ import annotations


from itertools import product
from pathlib import Path
from typing import Any
import logging

from sat_toolkit.formula import XorCNF, CNF, Truthtable
import numpy as np
import numpy.typing as npt

from ..cipher_model import SboxCipher, DifferentialCharacteristic
from .constants import apply_perm, connection_poly_4, connection_poly_8, do_mix_cols, do_shift_rows, expanded_rc, get_ddt, mixing_mat, tweakey_mask, tweakey_perm, update_tweakey
from .util import sbox8, sbox4, LfsrState


log = logging.getLogger(__name__)

contexts = []
interrupted = False

sbox8_dnf = Truthtable.from_indices(16, (sbox8.astype(np.uint16) << 0) | (np.arange(256, dtype=np.uint16) << 8))
sbox8_cnf = sbox8_dnf.to_cnf()

sbox4_dnf = Truthtable.from_indices(8, (sbox4.astype(np.uint16) << 0) | (np.arange(16, dtype=np.uint16) << 4))
sbox4_cnf = sbox4_dnf.to_cnf()

sboxes = {8: sbox8, 4: sbox4}
sbox_cnfs = {8: sbox8_cnf, 4: sbox4_cnf}

DDT4 = get_ddt(sbox4)
DDT8 = get_ddt(sbox8)

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
    ddt: np.ndarray[Any, np.dtype[np.uint16]] = DDT8
    block_size = 128

class Skinny64Characteristic(_SkinnyBaseCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint16]] = DDT4
    block_size = 64


class SkinnyBase(SboxCipher):
    sbox: np.ndarray
    ddt: np.ndarray
    block_size = 128
    key_size = 128
    tweak_size = 256
    sbox_bits = 8
    connection_poly: np.ndarray

    _tk2: np.ndarray[Any, np.dtype[np.int32]]
    _tk3: np.ndarray[Any, np.dtype[np.int32]]
    round_tweakeys: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: _SkinnyBaseCharacteristic, **kwargs):
        super().__init__(char, **kwargs)
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

    def _create_vars(self):

        self.add_index_array('sbox_in', (self.num_rounds + 1, 4, 4, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, 4, 4, self.sbox_bits))

        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

        self.ct = self.sbox_in[-1]
        self._fieldnames.add('ct')

        self._model_key_schedule()

    def _model_key_schedule(self):
        rnds = self.numrounds
        lfsr_updates = (self.numrounds - 1) // 2 + 1
        self.add_index_array('key', (4, 4, self.sbox_bits))

        self._fieldnames.add('tk2')
        self._fieldnames.add('tk3')
        self.add_index_array('_tk2', (4, 4, self.sbox_bits + lfsr_updates))
        self.add_index_array('_tk3', (4, 4, self.sbox_bits + lfsr_updates))


        tk2_tmp = self._tk2.reshape(16, self.sbox_bits + lfsr_updates)
        tk3_tmp = self._tk3.reshape(16, self.sbox_bits + lfsr_updates)

        self._tk2_lfsrs = [LfsrState(f'tk2_{i}', self.connection_poly[::-1].tolist(), tk2_tmp[i]) for i in range(16)]
        self._tk3_lfsrs = [LfsrState(f'tk3_{i}', self.connection_poly[::-1].tolist(), tk3_tmp[i]) for i in range(16)]


        round_tweakeys = []
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

            round_tweakeys.append((key, tk2, tk3))

        round_tweakeys = np.array(round_tweakeys, dtype=np.int32).reshape(rnds, 3, 4, 4, self.sbox_bits)

        self._round_tweakeys = round_tweakeys
        self._fieldnames.add('_round_tweakeys')

        self.add_index_array('round_tweakeys', (self.numrounds, 4, 4, self.sbox_bits))
        self.cnf += XorCNF.create_xor(round_tweakeys[:, 0].flatten(), round_tweakeys[:, 1].flatten(), round_tweakeys[:, 2].flatten(), self.round_tweakeys.flatten())

        self.tk2 = np.array(round_tweakeys[0][1]).reshape(4, 4, self.sbox_bits)
        self.tk3 = np.array(round_tweakeys[0][2]).reshape(4, 4, self.sbox_bits)
        self.tweak = np.array([self.tk2, self.tk3])
        self._fieldnames.add('tweak')

        # add lfsr model
        lfsr_cnf = CNF()
        for lfsr in self._tk2_lfsrs:
            for constraint in lfsr.get_constraints():
                lfsr_cnf += constraint.to_cnf()
        for lfsr in self._tk3_lfsrs:
            for constraint in lfsr.get_constraints():
                lfsr_cnf += constraint.to_cnf()

        self.cnf += lfsr_cnf



    def _model_linear_layer(self):
        for rnd in range(self.numrounds):
            in_rcs = expanded_rc[rnd]

            sb_mc_input = np.zeros((4, 4), object)
            for row, col in product(range(4), range(4)):
                sb_mc_input[row, col] = [self.sbox_out[rnd, row, col]]

            for row, col in product(range(2), range(4)):
                sb_mc_input[row, col].append(self.round_tweakeys[rnd, row, col])


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


class Skinny128(SkinnyBase):
    sbox = sbox8
    ddt = DDT8
    connection_poly = connection_poly_8
    block_size = 128
    key_size = 128
    tweak_size = 256
    sbox_bits = 8

class Skinny128LongKey(Skinny128):
    key_size: int = None # type: ignore
    tweak_size = 0

    def _model_key_schedule(self):
        self.add_index_array('round_tweakeys', (self.numrounds, 2, 4, self.sbox_bits))
        self.add_index_array('tweak', (0, ))
        self.key_size = self.round_tweakeys.size

        self.key = self.round_tweakeys.reshape(-1, self.sbox_bits)
        self._fieldnames.add('key')

class Skinny64(SkinnyBase):
    sbox = sbox4
    ddt = DDT4
    connection_poly = connection_poly_4
    block_size = 64
    key_size = 64
    tweak_size = 128
    sbox_bits = 4

class Skinny64LongKey(Skinny64):
    key_size: int = None # type: ignore
    tweak_size = 0

    def _model_key_schedule(self):
        self.add_index_array('round_tweakeys', (self.numrounds, 2, 4, self.sbox_bits))
        self.add_index_array('tweak', (0, ))
        self.key_size = self.round_tweakeys.size

        self.key = self.round_tweakeys.reshape(-1, self.sbox_bits)
        self._fieldnames.add('key')

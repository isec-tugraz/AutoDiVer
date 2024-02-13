#!/usr/bin/env python3
"""
This script tries to find a collision for RomulusHash based on a bitwise dual
characteristic in numpy's .npz file format.
"""
from __future__ import annotations
from skinny.util import sbox, LfsrState, get_solution_set
from skinny.constants import apply_perm, connection_poly, do_mix_cols, do_shift_rows, expanded_rc, get_ddt, mixing_mat, skinny_verbose, tweakey_mask, tweakey_perm, update_tweakey
from util import Model
from itertools import product
from pathlib import Path
import logging
from typing import Any, Literal
from binascii import hexlify
from sat_toolkit.formula import XorCNF, CNF, Truthtable
from cipher_model import SboxCipher, DifferentialCharacteristic
import numpy.typing as npt
import numpy as np
import zipfile
log = logging.getLogger(__name__)
contexts = []
interrupted = False
sbox_dnf = Truthtable.from_indices(16, (sbox.astype(np.uint16) << 0) | (np.arange(256, dtype=np.uint16) << 8))
sbox_cnf = sbox_dnf.to_cnf()
class SkinnyCharacteristic(DifferentialCharacteristic):
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
        return cls(sbox_in, sbox_out, tweakeys)
    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike, tweakeys: npt.ArrayLike):
        super().__init__(sbox_in, sbox_out)
        self.tweakeys = np.array(tweakeys, dtype=np.uint8)
        if self.tweakeys.shape != (self.num_rounds, 3, 4, 4):
            raise ValueError('tweakeys must have shape (num_rounds, 3, 4, 4)')
        # sanity check characteristic
        for i in range(len(self.tweakeys) - 1):
            assert np.all(self.tweakeys[i + 1] == update_tweakey(self.tweakeys[i])), f'tweakey update check failed at round {i}'
            rtk = np.bitwise_xor.reduce(self.tweakeys[i], axis=0) & tweakey_mask
            assert np.all(self.sbox_in[i + 1] == do_mix_cols(do_shift_rows(self.sbox_out[i] ^ rtk)))
        self.num_rounds = len(self.sbox_in)
class Skinny128(SboxCipher):
    sbox = sbox
    ddt = get_ddt(sbox)
    block_size = 128
    key_size = 128
    tweak_size = 256
    sbox_bits = 8
    _tk2: np.ndarray
    _tk3: np.ndarray
    def __init__(self, char: SkinnyCharacteristic):
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
        self.add_index_array('sbox_in', (self.num_rounds + 1, 4, 4, 8))
        self.add_index_array('sbox_out', (self.num_rounds, 4, 4, 8))
        self.pt = self.sbox_in[0]
        self.ct = self.sbox_in[-1]
        lfsr_updates = (self.numrounds - 1) // 2 + 1
        self.add_index_array('key', (4, 4, 8))
        self.add_index_array('_tk2', (4, 4, 8 + lfsr_updates))
        self.add_index_array('_tk3', (4, 4, 8 + lfsr_updates))
        self.tweak = np.array([self._tk2[..., :8], self._tk3[..., :8]])
        tk2_tmp = self._tk2.reshape(16, 8 + lfsr_updates)
        tk3_tmp = self._tk3.reshape(16, 8 + lfsr_updates)
        self._tk2_lfsrs = [LfsrState(f'tk2_{i}', connection_poly[::-1].tolist(), tk2_tmp[i]) for i in range(16)]
        self._tk3_lfsrs = [LfsrState(f'tk3_{i}', connection_poly[::-1].tolist(), tk3_tmp[i]) for i in range(16)]
        self.round_tweakeys = []
        for rnd in range(rnds):
            key = self.key.reshape(16, 8)
            # rows 2, 3 of the base tweakey are updated one round earlier
            # corresponding to indices in range(8, 16)
            # tk2 = [self._tk2_lfsrs[i].get_bit_range((rnd + (i in range(8, 16))) // 2) for i in range(16)]
            tk2 = [self._tk2_lfsrs[i].get_bit_range(len(self._tk2_lfsrs[i].vars) - 8 - (rnd + (i in range(8, 16))) // 2) for i in range(16)]
            tk3 = [self._tk3_lfsrs[i].get_bit_range((rnd + (i in range(8, 16))) // 2) for i in range(16)]
            # tk3 = [self._tk3_lfsrs[i].get_bit_range((rnds - 1) // 2 - (rnd + (i in range(8, 16))) // 2) for i in range(16)]
            key = apply_perm(key, tweakey_perm, rnd)
            tk2 = apply_perm(tk2, tweakey_perm, rnd)
            tk3 = apply_perm(tk3, tweakey_perm, rnd)
            self.round_tweakeys.append((key, tk2, tk3))
        self.round_tweakeys = np.array(self.round_tweakeys).reshape(rnds, 3, 4, 4, 8)
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
                    sb_out_var = self.sbox_in[rnd + 1][row, col]
                    sb_in_vars = sb_mc_input[mixing_mat[row] != 0, col]
                    sb_in_vars = sum(sb_in_vars, start=[])
                    constant = np.bitwise_xor.reduce(in_rcs[mixing_mat[row] != 0, col])
                    constant = np.unpackbits(constant, bitorder='little')
                    lin_layer_cnf += XorCNF.create_xor(sb_out_var, *sb_in_vars, rhs=constant.astype(np.int32))
            self.cnf += lin_layer_cnf
    @staticmethod
    def _get_cnf(delta_in: int, delta_out: int) -> CNF:
        if delta_in == delta_out == 0:
            return sbox_cnf
        in_vals = get_solution_set(delta_in, delta_out)
        dnf = Truthtable.from_indices(16, (sbox[in_vals].astype(np.uint16) << 0) | (in_vals.astype(np.uint16) << 8))
        cnf = dnf.to_cnf()
        return cnf
    def _model_sboxes(self):
        for rnd in range(self.numrounds):
            for row, col in product(range(4), range(4)):
                sbox_in = self._sbox_in[rnd, row, col]
                sbox_out = self._sbox_out[rnd, row, col]
                cnf = self._get_cnf(sbox_in, sbox_out)
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
def find_collision(bit_char_file: str, start_arg: int, to_arg: int, iv_arg: str, tid: int):
    suffix = '_coll' if iv_arg == 'free' else '_full_coll'
    out_file = bit_char_file.replace('.npz', f'{suffix}.npz')
    log_file = bit_char_file.replace('.npz', f'{suffix}.log')
    with np.load(bit_char_file) as f:
        sbox_in = f.get('sbox_in')[start_arg:to_arg]
        sbox_out = f.get('sbox_out')[start_arg:to_arg]
        tweakeys = f.get('tweakeys')[start_arg:to_arg]
    char = SkinnyCharacteristic(sbox_in, sbox_out, tweakeys)
    skinny = Skinny128(char)
    numrounds = skinny.numrounds
    m = skinny.solve()
    # sanity check result
    for i in range(numrounds):
        sbi = skinny.sbox_input_var[i].get_value(m)
        sbo = skinny.sbox_output_var[i].get_value(m)
        assert np.all(sbo == sbox[sbi])
        assert np.all(sbox[sbi] ^ sbox[sbi ^ sbox_in[i]] == sbox_out[i])
        # assert np.all(sbox[sbi] ^ sbox[sbi ^ tbox_in[i]] == tbox_out[i])
        # assert np.all(sbox[sbi ^ sbox_in[i]] ^ sbox[sbi  ^ sbox_in[i] ^ tbox_in[i]] == tbox_out[i])
        if i + 1 in range(numrounds):
            assert np.all(update_tweakey(skinny.get_rtk_value(m, i)) == skinny.get_rtk_value(m, i + 1))
            rtk = np.bitwise_xor.reduce(skinny.get_rtk_value(m, i), axis=0) & tweakey_mask
            rc = expanded_rc[i]
            sbo = skinny.sbox_output_var[i].get_value(m)
            sbi = skinny.sbox_input_var[i + 1].get_value(m)
            assert np.all(sbi == do_mix_cols(do_shift_rows(sbo ^ rtk ^ rc)))
    pt = skinny.pt.get_value(m)
    assert np.all(pt == skinny.sbox_input_var[0].get_value(m))
    key = skinny.key.get_value(m)
    m1 = skinny.m1.get_value(m)
    m2 = skinny.m2.get_value(m)
    # coll = np.array([pt, key, m1, m2])
    pt_delta = np.array([1] + [0] * 15, dtype=np.uint8).reshape(4, 4)
    a = np.zeros([2, numrounds + 1, 4, 4], np.uint8)
    b = np.zeros([2, numrounds + 1, 4, 4], np.uint8)
    a[0], t1 = skinny_verbose(pt, np.array([key, m1, m2]), numrounds)
    a[1], _ = skinny_verbose(pt ^ pt_delta, np.array([key, m1, m2]), numrounds)
    b[0], t2 = skinny_verbose(pt, np.array([key, m1, m2]) ^ tweakeys[0], numrounds)
    b[1], _ = skinny_verbose(pt ^ pt_delta, np.array([key, m1, m2]) ^ tweakeys[0], numrounds)
    a = a.transpose(1, 0, 2, 3)
    b = b.transpose(1, 0, 2, 3)
    coll_tweakeys_1 = np.array([key, m1, m2])
    coll_tweakeys_2 = np.array([key, m1, m2]) ^ tweakeys[0]
    lr = bytes(pt) + bytes(coll_tweakeys_1[0])
    m1 = bytes(coll_tweakeys_1[1:])
    m2 = bytes(coll_tweakeys_2[1:])
    log.info(f'difference after {numrounds} rounds: {hexlify(bytes((a[numrounds] ^ b[numrounds]))).decode()}')
    log.info(f'lr  = unhexlify(b"{hexlify(lr).decode()}")')
    log.info(f'm1  = unhexlify(b"{hexlify(m1).decode()}")')
    log.info(f'm2  = unhexlify(b"{hexlify(m2).decode()}")')
    with open(log_file, 'a') as f:
        f.write(f'args: {vars(args)}\n\n')
        f.write(f'using characteristic {bit_char_file} for {numrounds} rounds\n')
        f.write(f'difference after {numrounds} rounds: ')
        f.write(hexlify(bytes((a[numrounds] ^ b[numrounds]))).decode() + '\n')
        f.write(f'L||R: {hexlify(lr).decode()}\n')
        f.write(f'M1:   {hexlify(m1).decode()}\n')
        f.write(f'M2:   {hexlify(m2).decode()}\n\n')
    np.savez(out_file, args=vars(args), numrounds=numrounds, execution_a=a, execution_b=b, lr=lr, m1=m1, m2=m2)
    with zipfile.ZipFile(out_file, mode='a') as zf:
        zf.write(__file__, path.basename(__file__))
        zf.write(bit_char_file)
    assert np.all(a[numrounds] == b[numrounds])
    assert m1 != m2 and romulush_reduce(lr, m1, numrounds) == romulush_reduce(lr, m2, numrounds)
    # embed()
def main(bit_char_file: str):
    find_collision(bit_char_file, args.start, args.to, args.iv, 0)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bit_char_file', type=str, help='differential characteristic in .npz format')
    parser.add_argument('--iv', choices=['free', 'zero', 'random', 'random_once'], default='free', help='constrain the iv to either zero, a random iv returned by the compression function, or not contain it')
    parser.add_argument('--measure-prefix-filter', nargs='?', default=0, const=1000, help='Measure the effect of filtering prefixes based on first 2 rounds')
    parser.add_argument('--from', type=int, default=None, dest='start', help='start model in round 0..numrounds-1')
    parser.add_argument('--to', type=int, default=None, help='stop model in round 1..numrounds (exclusive)')
    args = parser.parse_args()
    main(args.bit_char_file)
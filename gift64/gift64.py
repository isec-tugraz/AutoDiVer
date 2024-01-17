#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import sys
import argparse
from copy import copy
from random import randint
import numpy as np
from typing import Any
from sat_toolkit.formula import CNF
sys.path.append('../')
from gift_util import bit_perm, P64, DDT as GIFT_DDT, GIFT_RC, pack_bits, unpack_bits
from cipher_model import SboxCipher, DifferentialCharacteristic
from pyximport import install
install()
from gift_cipher import gift64_enc
class Gift64(SboxCipher):
    cipher_name = "GIFT64"
    sbox = np.array([int(x, 16) for x in "1a4c6f392db7508e"], dtype=np.uint8)
    block_size = 64
    key_size = 128
    sbox_bits = 4
    sbox_count = 16
    key: np.ndarray[Any, np.dtype[np.int32]]
    def __init__(self, char: DifferentialCharacteristic):
        super().__init__(char)
        self.char = char
        self.num_rounds = char.num_rounds
        assert self.char.sbox_in.shape == self.char.sbox_out.shape
        if self.char.sbox_in.shape != self.char.sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')
        for i in range(1, self.num_rounds):
            lin_input = self.char.sbox_out[i - 1]
            lin_output = self.char.sbox_in[i]
            permuted = bit_perm(lin_input)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')
        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('key', (self.sbox_count*2, self.sbox_bits))
        self._model_sboxes()
        self._model_key_schedule()
        self._model_linear_layer()
    def applyPerm(self, array: np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.int32]]:
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[P64]
        arrayOut = arrayPermuted.reshape(16, 4)
        return arrayOut
    def _model_key_schedule(self) -> None:
        keyWords = self.key.copy().reshape(8, 16)
        RK = []
        for _ in range(self.num_rounds):
            rk = np.empty(len(keyWords[0]) + len(keyWords[1]), dtype=keyWords.dtype)
            rk[0::2] = keyWords[0]
            rk[1::2] = keyWords[1]
            keyWords[0] = np.roll(keyWords[0], -12)
            keyWords[1] = np.roll(keyWords[1], -2)
            #rotatate the words by 2
            keyWords = np.roll(keyWords, -2, axis=0)
            rk = rk.reshape(16, 2)
            RK.append(rk)
        self._round_keys = np.array(RK)
    def _addKey(self, Y, X, K, RC: int) -> None:
        """
        Y = addKey(X, K)
        """
        X = X.copy()
        X_flat = X.reshape(-1) # don't use .flatten() here because it creates a copy
        # flip bits according to round constant
        X_flat[3]  *= (-1)**(RC & 0x1)
        X_flat[7]  *= (-1)**((RC >> 1) & 0x1)
        X_flat[11] *= (-1)**((RC >> 2) & 0x1)
        X_flat[15] *= (-1)**((RC >> 3) & 0x1)
        X_flat[19] *= (-1)**((RC >> 4) & 0x1)
        X_flat[23] *= (-1)**((RC >> 5) & 0x1)
        X_flat[63] *= (-1)
        key_xor_cnf = CNF()
        key_xor_cnf += CNF.create_all_equal(X[:, 2:].flatten(), Y[:, 2:].flatten())
        key_xor_cnf += CNF.create_xor(X[:, :2].flatten(), Y[:, :2].flatten(), K.flatten(), )
        self.cnf += key_xor_cnf
    def _model_linear_layer(self) -> None:
        cnf = CNF()
        for r in range(self.num_rounds):
            permOut = self.applyPerm(self.sbox_out[r])
            self._addKey(permOut, self.sbox_in[r+1], self._round_keys[r], GIFT_RC[r])
        self.cnf += cnf
def sanity_check_gift():
    numrounds = 26
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = DifferentialCharacteristic(sbi, sbo)
    gift = Gift64(char)
    for bit_var in gift.key.flatten():
        gift.cnf.append([bit_var * (-1)**randint(0,1)])
    for bit_var in gift.sbox_in[0].flatten():
        gift.cnf.append([bit_var * (-1)**randint(0,1)])
    model = gift.solve()
    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)
    for r, round_sbi in enumerate(sbi):
        ref = gift64_enc(sbi[0], key, r)
        assert np.all(round_sbi == ref)
    mantissa, exponent = gift._count_solutions(verbosity=0)
    assert mantissa * 2**exponent == 1
    print("sanity check 1 passed")
    char = (
        ("0000000c00000006", "0000000200000002"),
        ("0000000002020000", "0000000005050000"),
        ("0000005000000050", "0000002000000020"),
        ("0000000000000202", "0000000000000505"),
        ("0000000500000005", "0000000200000002"),
        ("0000000002020000", "0000000005050000"),
        ("0000005000000050", "0000002000000020"),
        ("0000000000000202", "0000000000000505"),
        ("0000000500000005", "0000000f0000000f"),
    )
    sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
    sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
    gift = Gift64(char)
    model = gift.solve()
    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    for r, round_sbi in enumerate(sbi):
        ref = gift64_enc(sbi[0], key, r)
        ref_xor = gift64_enc(sbi[0] ^ sbi_delta[0], key, r)
        assert np.all(round_sbi == ref)
        if r < gift.num_rounds - 1:
            assert np.all(round_sbi ^ sbi_delta[r] == ref_xor)
    print('sanity check 2 passed')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trail', help='Text file containing the sbox input and output differences.\n'\
                                      'Input and output differences are listed on separate lines.')
    args = parser.parse_args()
    trail = []
    with open(args.trail, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            assert len(line) == 16
            line_deltas = [int(l, 16) for l in line[::-1]]
            trail.append(line_deltas)
    trail = np.array(trail)
    if len(trail) % 2 != 0:
        print(f'expected an even number of differences in {args.trail!r}')
        raise SystemExit(1)
    sbox_in = trail[0::2]
    sbox_out = trail[1::2]
    char = DifferentialCharacteristic(sbox_in, sbox_out)
    ddt_prob = char.log2_ddt_probability(GIFT_DDT)
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    sanity_check_gift()
    gift = Gift64(char)
    # gift.count_key_space()
    for _ in range(10):
        gift.count_probability_for_random_key(verbosity=0)
    gift.count_probability(verbosity=0)
    from IPython import embed; embed()
if __name__ == "__main__":
    raise SystemExit(main())
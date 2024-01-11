#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import sys
import argparse
import numpy as np
from typing import Any
from sat_toolkit.formula import CNF
sys.path.append('../')
from gift_util import bit_perm
from cipher_model import SboxCipher, DifferentialCharacteristic
P64 = np.array((0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3,
                4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,
                8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
                12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15))
ddt = np.array([[16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  2,  2,  0,  2,  2,  2,  2,  2,  0,  0,  2],
                [ 0,  0,  0,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  2,  2,  0],
                [ 0,  0,  0,  0,  0,  2,  2,  0,  2,  0,  0,  2,  2,  2,  2,  2],
                [ 0,  0,  0,  2,  0,  4,  0,  6,  0,  2,  0,  0,  0,  2,  0,  0],
                [ 0,  0,  2,  0,  0,  2,  0,  0,  2,  0,  0,  0,  2,  2,  2,  4],
                [ 0,  0,  4,  6,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  2,  0],
                [ 0,  0,  2,  0,  0,  2,  0,  0,  2,  2,  2,  4,  2,  0,  0,  0],
                [ 0,  0,  0,  4,  0,  0,  0,  4,  0,  0,  0,  4,  0,  0,  0,  4],
                [ 0,  2,  0,  2,  0,  0,  2,  2,  2,  0,  2,  0,  2,  2,  0,  0],
                [ 0,  4,  0,  0,  0,  0,  4,  0,  0,  2,  2,  0,  0,  2,  2,  0],
                [ 0,  2,  0,  2,  0,  0,  2,  2,  2,  2,  0,  0,  2,  0,  2,  0],
                [ 0,  0,  4,  0,  4,  0,  0,  0,  2,  0,  2,  0,  2,  0,  2,  0],
                [ 0,  2,  2,  0,  4,  0,  0,  0,  0,  0,  2,  2,  0,  2,  0,  2],
                [ 0,  4,  0,  0,  4,  0,  0,  0,  2,  2,  0,  0,  2,  2,  0,  0],
                [ 0,  2,  2,  0,  4,  0,  0,  0,  0,  2,  0,  2,  0,  0,  2,  2]],
               dtype=np.uint8)
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
        self.trail_sbox_in = np.array(sbox_in)
        self.trail_sbox_out = np.array(sbox_out)
        assert self.trail_sbox_in.shape == self.trail_sbox_out.shape
        self.num_rounds = len(self.trail_sbox_in)
        if self.trail_sbox_in.shape != self.trail_sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')
        for i in range(1, self.num_rounds):
            lin_input = self.trail_sbox_out[i - 1]
            lin_output = self.trail_sbox_in[i]
            permuted = bit_perm(lin_input)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')
        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('key', (1, self.sbox_count*2, self.sbox_bits))
        self._model_sboxes()
        self._model_key_schedule()
        self._model_linear_layer()
    def applyPerm(self, array: np.ndarray):
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[P64]
        arrayOut = arrayPermuted.reshape(16, 4)
        return arrayOut
    def _model_key_schedule(self):
        keyWords = self.key.copy().reshape(8, 16)
        # print(f'{keyWords=}')
        RK = []
        for _ in range(self.num_rounds):
            rk = np.empty(len(keyWords[0]) + len(keyWords[1]), dtype=keyWords.dtype)
            rk[0::2] = keyWords[0]
            rk[1::2] = keyWords[1]
            # print(f'{rk=}')
            # print(f'{keyWords[0]=}')
            # print(f'{keyWords[1]=}')
            keyWords[0] = np.roll(keyWords[0], -12)
            keyWords[1] = np.roll(keyWords[1], -2)
            # print(f'{keyWords[0]=}')
            # print(f'{keyWords[1]=}')
            #rotatate the words by 2
            # print(f'{keyWords=}')
            keyWords = np.roll(keyWords, -2, axis=0)
            # print(f'{keyWords=}')
            rk = rk.reshape(16, 2)
            RK.append(rk)
        self._round_keys = np.array(RK)
    def _addKey(self, Y, X, K):
        """
        Y = addKey(X, K)
        """
        key_xor_cnf = CNF()
        key_xor_cnf += CNF.create_all_equal(X[:, 2:].flatten(), Y[:, 2:].flatten())
        key_xor_cnf += CNF.create_xor(X[:, :2].flatten(), Y[:, :2].flatten(), K.flatten())
        return key_xor_cnf
    def _model_linear_layer(self):
        cnf = CNF()
        for r in range(self.num_rounds):
            #Permutation Layer: no permutation for last round
            if r != self.num_rounds - 1:
                permOut = self.applyPerm(self.sbox_out[r])
            else:
                permOut = self.sbox_in[r]
            cnf += self._addKey(permOut, self.sbox_in[r+1], self._round_keys[r])
        self.cnf += cnf
if __name__ == "__main__":
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
    rounds = len(trail) // 2
    sbox_list = [int(x, 16) for x in '1a4c6f392db7508e']
    sbox_in = trail[0::2]
    sbox_out = trail[1::2]
    char = DifferentialCharacteristic(sbox_in, sbox_out)
    ddt_prob = char.log2_ddt_probability(ddt)
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    gift = Gift64(char)
    print('constructed model')
    gift.count()
#!/usr/bin/env python
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations
import sys
import argparse
from math import log2
import numpy as np
import numpy.typing as npt
from typing import Any
from pyapproxmc import Counter
sys.path.append('../')
from model_util import *
from util import IndexSet
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
class CipherModel(IndexSet):
    _sboxIn: np.ndarray[Any, np.dtype[np.int32]]
    _sboxOut: np.ndarray[Any, np.dtype[np.int32]]
    MK: np.ndarray[Any, np.dtype[np.int32]]
    def __init__(self, sboxList: npt.ArrayLike, blockSize: int, sboxSize: int, nRound: int, nSbox: int, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike):
        super().__init__()
        self._sbox = np.array(sboxList)
        self._sboxSize = sboxSize
        self._blockSize = blockSize
        self._nRound = nRound
        self._nSbox = nSbox
        self.trail_sbox_in = np.array(sbox_in)
        self.trail_sbox_out = np.array(sbox_out)
        if self.trail_sbox_in.shape != self.trail_sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')
        #generate Variables
        self.add_index_array('_sboxIn', (self._nRound+1, self._nSbox, self._sboxSize))
        self.add_index_array('_sboxOut', (self._nRound, self._nSbox, self._sboxSize))
        self.add_index_array('MK', (1, self._nSbox*2, self._sboxSize))
        self._rk = self._keySchedule()
        # print(self._rk)
        # test permutation
        # temp = self.applyPerm(self._sboxIn[0])
        # print(self._sboxIn[0])
        # print(temp)
        self.cnf = self.genCnf()
        # print(self._completeCnf)
    def applyPerm(self, array: np.ndarray):
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[P64]
        arrayOut = arrayPermuted.reshape(16, 4)
        return arrayOut
    def _keySchedule(self):
        keyWords = self.MK.flatten()
        keyWords = keyWords.reshape(8, 16)
        # print(f'{keyWords=}')
        RK = []
        for r in range(self._nRound):
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
        return RK
    def addKey(self, Y, X, K):
        """
        Y = addKey(X, K)
        """
        CC = CNF()
        for s in range(self._nSbox):
            #Key bits are added in the bit position 0 and 1 of each sbox
            var = [0]
            var.append(Y[s][0])
            var.append(K[s][0])
            var.append(X[s][0])
            CC1 = xorModel(var)
            CC += CC1
            var = [0]
            var.append(Y[s][1])
            var.append(K[s][1])
            var.append(X[s][1])
            CC1 = xorModel(var)
            CC += CC1
            var = [0]
            var.append(Y[s][2])
            var.append(X[s][2])
            CC1 = eqModel(var)
            CC += CC1
            var = [0]
            var.append(Y[s][3])
            var.append(X[s][3])
            CC1 = eqModel(var)
            CC += CC1
        return CC
    def sboxLayer(self, X, Y, inDiff, outDiff):
        """
        Y = S(X)
        """
        CC = CNF()
        for s in range(self._nSbox):
            var = [0]
            var += list(X[s])
            var += list(Y[s])
            # print(f'{var = }')
            CC1 = sboxModel(self._sbox, self._sboxSize, self._sboxSize,\
                    inDiff[s], outDiff[s], var)
            CC += CC1
        return CC
    def genCnf(self):
        cnf = CNF()
        for r in range(0, self._nRound):
            #Sbox Layer
            cnf += self.sboxLayer(self._sboxIn[r], self._sboxOut[r],\
                    self.trail_sbox_in[r], self.trail_sbox_out[r])
            #Permutation Layer: no permutation for last round
            if(r != (self._nRound - 1)):
                permOut = self.applyPerm(self._sboxOut[r])
            else:
                permOut = self._sboxOut[r]
            cnf += self.addKey(permOut, self._sboxIn[r+1], self._rk[r])
        return cnf
    def solve(self):
        counter = Counter()
        # for clause in self.cnf:
            # counter.add_clause(clause)
        counter.add_clauses(list(self.cnf))
        mantissa, exponent = counter.count()
        print(f'{mantissa} * 2**{exponent} solutions')
        log2_prob = (log2(mantissa) + exponent) - (128 + 64)
        print(f'probability : 2**{log2_prob:.2f}')
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
            line_deltas = [int(l, 16) for l in line]
            trail.append(line_deltas)
    trail = np.array(trail)
    if len(trail) % 2 != 0:
        print(f'expected an even number of differences in {args.trail!r}')
        raise SystemExit(1)
    rounds = len(trail) // 2
    sbox_list = [int(x, 16) for x in '1a4c6f392db7508e']
    sbox_in = trail[0::2]
    sbox_out = trail[1::2]
    ddt_prob = np.log2(ddt[sbox_in, sbox_out] / 16).sum()
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    gift = CipherModel(sbox_list, 64, 4, rounds, 16, sbox_in, sbox_out)
    gift.solve()
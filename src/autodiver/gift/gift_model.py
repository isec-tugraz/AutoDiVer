#!/usr/bin/env python3
"""
model the solutions of a differential characteristic for GIFT64 and count them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from io import StringIO

import numpy as np
from typing import Any
from sat_toolkit.formula import XorCNF

from .gift_util import bit_perm, P64, P128, DDT as GIFT_DDT, GIFT_RC
from ..cipher_model import SboxCipher, DifferentialCharacteristic

log = logging.getLogger(__name__)

_PREAMBLE = r"""
\documentclass{standalone}
\usepackage{tikz}
\usepackage{gift}
\usepackage{tugcolors}
\begin{document}

""".strip("\n")

_DOCUMENT_END = r"""
\end{document}
""".strip("\n")



class _GiftCharacteristic(DifferentialCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint8]] = GIFT_DDT

    num_sboxes: int
    block_size: int
    permutation: np.ndarray

    _round_keys: np.ndarray[Any, np.dtype[np.int32]]

    @classmethod
    def verify(cls, sbox_in: np.ndarray, sbox_out: np.ndarray) -> None:
        if len(sbox_in.shape) != 2:
            raise ValueError(f'sbox_in must have 2 dimensions (round, sbox), sbxo_in.shape = {sbox_in.shape}')

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')

        if sbox_in.shape[1] != cls.num_sboxes:
            raise ValueError(f'sbox_in.shape[1] must be {cls.num_sboxes}')

        num_rounds = sbox_in.shape[0]

        for i in range(1, num_rounds):
            lin_input = sbox_out[i - 1]
            lin_output = sbox_in[i]
            permuted = bit_perm(lin_input, cls.permutation)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')


    @classmethod
    def load_txt(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        trail = []
        with open(characteristic_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                assert len(line) == cls.num_sboxes, f'expected {cls.num_sboxes} sboxes, got {len(line)}'
                line_deltas = [int(l, 16) for l in line[::-1]]
                trail.append(line_deltas)

        trail = np.array(trail)
        if len(trail) % 2 != 0:
            log.error(f'expected an even number of differences in {characteristic_path!r}')
            raise ValueError(f'expected an even number of differences in {characteristic_path!r}')

        sbox_in = trail[0::2]
        sbox_out = trail[1::2]

        cls.verify(sbox_in, sbox_out)
        return cls(sbox_in, sbox_out, file_path=characteristic_path)

    @classmethod
    def load_np(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']

        cls.verify(sbox_in, sbox_out)
        return cls(sbox_in, sbox_out, file_path=characteristic_path)


    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        if characteristic_path.suffix == '.txt':
            return cls.load_txt(characteristic_path)
        elif characteristic_path.suffix == '.npz':
            return cls.load_np(characteristic_path)
        else:
            raise ValueError(f'unsupported file type {characteristic_path.suffix}')

    def tikzify(self) -> str:
        result = StringIO()

        inv_perm = np.zeros_like(self.permutation)
        inv_perm[self.permutation] = np.arange(len(self.permutation), dtype=self.permutation.dtype)


        print(_PREAMBLE, file=result)
        print(r"""
  \begin{tikzpicture}[gift, xscale=.8, sbox/.append style={minimum size=.4cm}, op/.append style={scale=.8}]
              """.strip("\n").rstrip(), file=result)

        cmd = "smallgiftfalse" if self.block_size == 128 else "smallgifttrue"
        print(f"      \\{cmd}", file=result)
        print(f"      \\giftinit[i]", file=result)
        print(f"      \\spnlinktrue", file=result)

        num_rounds = len(self.sbox_in)
        for rnd in range(num_rounds):
            print(f"      \\giftround", file=result)

            active_sboxes, = np.nonzero(self.sbox_in[rnd])
            active_input_bits = []
            active_output_bits = []
            for nibble_idx in range(self.num_sboxes):
                for bit_idx in range(4):
                    if self.sbox_in[rnd, nibble_idx] & (1 << bit_idx):
                        active_input_bits.append(4 * nibble_idx + bit_idx)
                    if self.sbox_out[rnd, nibble_idx] & (1 << bit_idx):
                        active_output_bits.append(4 * nibble_idx + bit_idx)

            active_sboxes_str = ",".join(str(x) for x in active_sboxes)
            active_input_bits_str = ",".join(str(x) for x in active_input_bits)
            active_bit_transitions = ",".join(f"{x}/{inv_perm[x]}" for x in active_output_bits)
            print(f"      \\giftmarkbits[tugblue]{{{active_input_bits_str}}}{{{active_sboxes_str}}}{{{active_bit_transitions}}}", file=result)

        print(f"      \\giftfini", file=result)
        print(r"      \foreach \i in {0,4,...,\bits} { \draw (b\i|-here) node[below,gray,inner sep=0pt,font=\tiny] {\i}; }", file=result)



        print(r"""
  \end{tikzpicture}
        """.strip("\n").rstrip(), file=result)

        print(_DOCUMENT_END, file=result)
        return result.getvalue()


class Gift64Characteristic(_GiftCharacteristic):
    num_sboxes = 16
    block_size = 64
    permutation = P64

class Gift128Characteristic(_GiftCharacteristic):
    num_sboxes = 32
    block_size = 128
    permutation = P128


class _Gift(SboxCipher):
    sbox = np.array([int(x, 16) for x in "1a4c6f392db7508e"], dtype=np.uint8)
    ddt  = GIFT_DDT
    key_size = 128
    sbox_bits = 4
    permutation: np.ndarray

    sbox_count: int
    characteristic_type: type

    _round_keys: np.ndarray[Any, np.dtype[np.int32]]

    def __init__(self, char: DifferentialCharacteristic, **kwargs):
        if not isinstance(char, self.characteristic_type):
            raise ValueError(f'expected {self.characteristic_type}, got {type(char)}')

        super().__init__(char, **kwargs)
        self.char = char
        self.num_rounds = char.num_rounds

        #generate Variables
        self.add_index_array('sbox_in', (self.num_rounds+1, self.sbox_count, self.sbox_bits))
        self.add_index_array('sbox_out', (self.num_rounds, self.sbox_count, self.sbox_bits))
        self.add_index_array('key', (32, self.sbox_bits))

        self.add_index_array('tweak', (0,))
        self.pt = self.sbox_in[0]
        self._fieldnames.add('pt')

        self._model_sboxes()
        self._model_key_schedule()
        self._model_linear_layer()

        self.cnf.nvars = self.numvars

    def applyPerm(self, array: np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.int32]]:
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[self.permutation]
        arrayOut = arrayPermuted.reshape(self.sbox_count, 4)
        return arrayOut

    def _model_key_schedule(self) -> None:
        raise NotImplementedError("implement in subclass")

    def _addKey(self, Y, X, K, RC: int) -> None:
        raise NotImplementedError("implement in subclass")

    def _model_linear_layer(self) -> None:
        for r in range(self.num_rounds):
            permOut = self.applyPerm(self.sbox_out[r])
            self._addKey(permOut, self.sbox_in[r+1], self._round_keys[r], GIFT_RC[r])

class Gift128(_Gift):
    cipher_name = "GIFT128"
    characteristic_type = Gift128Characteristic
    block_size = 128
    sbox_count = 32
    permutation = P128

    def _model_key_schedule(self) -> None:
       keyWords = self.key.copy().reshape(8, 16)
       RK = []
       for _ in range(self.num_rounds):
           keyWords32 = keyWords.copy().reshape(4, 32)
           rk = np.empty(len(keyWords32[0]) + len(keyWords32[2]), dtype=keyWords.dtype)
           rk[0::2] = keyWords32[0]
           rk[1::2] = keyWords32[2]
           # print(keyWords32[0])
           # print(keyWords32[2])
           # print(rk)
           # print(rk.shape)

           keyWords[0] = np.roll(keyWords[0], -12)
           keyWords[1] = np.roll(keyWords[1], -2)

           #rotatate the words by 2
           keyWords = np.roll(keyWords, -2, axis=0)
           rk = rk.reshape(32, 2)
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
        X_flat[127] *= (-1)

        key_xor_cnf = XorCNF()
        key_xor_cnf += XorCNF.create_xor(X[:, :1].flatten(), Y[:, :1].flatten())
        key_xor_cnf += XorCNF.create_xor(X[:, 3:].flatten(), Y[:, 3:].flatten())
        key_xor_cnf += XorCNF.create_xor(X[:, 1:3].flatten(), Y[:, 1:3].flatten(), K.flatten())
        self.cnf += key_xor_cnf


class Gift64(_Gift):
    cipher_name = "GIFT64"
    characteristic_type = Gift64Characteristic
    block_size = 64
    sbox_count = 16
    permutation = P64

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

        key_xor_cnf = XorCNF()
        key_xor_cnf += XorCNF.create_xor(X[:, 2:].flatten(), Y[:, 2:].flatten())
        key_xor_cnf += XorCNF.create_xor(X[:, :2].flatten(), Y[:, :2].flatten(), K.flatten())
        self.cnf += key_xor_cnf

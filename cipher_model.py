"""
Cipher model base classes
"""
from __future__ import annotations
from util import IndexSet
from math import log2
import numpy as np
import numpy.typing as npt
from sat_toolkit.formula import CNF, Truthtable
from pyapproxmc import Counter
from pycryptosat import Solver
from util import IndexSet
from typing import Any
class DifferentialCharacteristic():
    num_rounds: int
    sbox_in: np.ndarray[Any, np.dtype[np.int8]]
    sbox_out: np.ndarray[Any, np.dtype[np.int8]]
    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike):
        self.sbox_in = np.array(sbox_in, dtype=np.int8)
        self.sbox_out = np.array(sbox_out, dtype=np.int8)
        if self.sbox_in.shape != self.sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')
        self.num_rounds = len(self.sbox_in)
    def log2_ddt_probability(self, ddt: np.ndarray):
        ddt_prob = np.log2(ddt[self.sbox_in, self.sbox_out] / len(ddt)).sum()
        return ddt_prob
class SboxCipher(IndexSet):
    cipher_name: str
    sbox: np.ndarray[Any, np.dtype[np.uint8]]
    block_size: int
    key_size: int
    sbox_bits: int
    num_rounds: int
    sbox_in: np.ndarray[Any, np.dtype[np.int32]]
    sbox_out: np.ndarray[Any, np.dtype[np.int32]]
    cnf: CNF
    def __init__(self, char: DifferentialCharacteristic):
        super().__init__()
        self.char = char
        self.num_rounds = char.num_rounds
        self.cnf = CNF()
    def get_solution_set_cnf(self, delta_in, delta_out):
        x = np.arange(len(self.sbox), dtype=np.uint8)
        x_set, = np.where(self.sbox[x] ^ self.sbox[x ^ delta_in] == delta_out)
        lut = np.zeros((len(self.sbox), len(self.sbox)), dtype=np.uint8)
        lut[x_set, self.sbox[x_set]] = 1
        dnf = Truthtable.from_lut(lut.T.flatten())
        cnf = dnf.to_cnf()
        return cnf
    def _model_sboxes(self):
        # assert self.sbox_in.shape == self.sbox_out.shape
        # assert self.sbox_in.shape == self.char.sbox_in.shape
        inp = self.sbox_in.reshape(-1, self.sbox_bits)
        out = self.sbox_out.reshape(-1, self.sbox_bits)
        delta_in = self.char.sbox_in.reshape(-1)
        delta_out = self.char.sbox_out.reshape(-1)
        sbox_cnf = CNF()
        for inp, out, delta_in, delta_out in zip(inp, out, delta_in, delta_out):
            mapping = np.concatenate((np.array([0], dtype=np.int32), inp, out))
            cnf = self.get_solution_set_cnf(delta_in, delta_out).translate(mapping)
            sbox_cnf += cnf
        self.cnf += sbox_cnf
    def _model_linear_layer(self):
        raise NotImplementedError("this should be implemented by subclasses")
    def _solve(self):
        solver = Solver()
        solver.add_clauses(self.cnf)
        is_sat, model = solver.solve()
        if not is_sat:
            raise ValueError('cnf is UNSAT')
        return model
    def count(self):
        counter = Counter()
        counter.add_clauses(self.cnf)
        mantissa, exponent = counter.count()
        print(f'{mantissa} * 2**{exponent} solutions')
        log2_prob = (log2(mantissa) + exponent) - (self.block_size + self.key_size)
        print(f'probability : 2**{log2_prob:.2f}')
        return log2_prob
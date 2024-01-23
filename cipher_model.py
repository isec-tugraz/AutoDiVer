"""
Cipher model base classes
"""
from __future__ import annotations
from util import IndexSet
from math import log2
import copy
import os
import logging
import time
import subprocess as sp
import tempfile
import sys
import numpy as np
import numpy.typing as npt
from sat_toolkit.formula import CNF, Truthtable
from pycryptosat import Solver
from util import IndexSet, Model
from typing import Any
log = logging.getLogger('main')
def count_solutions(cnf: CNF, epsilon: float, delta: float, verbosity: int=2, sampling_set: list[int] | None=None) -> int:
    sampling_set_log = " over {len(sampling_set)} variables" if sampling_set is not None else ""
    log.info(f'counting solutions to cnf with {cnf.nvars} variables and {len(cnf)} clauses{sampling_set_log}, {epsilon=}, {delta=}')
    # create temporary file for cnf
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf') as f:
        if sampling_set is not None:
            sampling_set_str = ' '.join(str(x) for x in sampling_set) + ' 0'
            f.write(f'c ind {sampling_set_str}\n')
        f.write(cnf.to_dimacs())
        f.flush()
        # run approxmc
        seed = int.from_bytes(os.urandom(4), 'little')
        args = ['approxmc', f'--seed={seed}', f'-e{epsilon}', f'-d{delta}', '--sparse=1', f'-v{verbosity}', f.name]
        log.info(f'running: {" ".join(args)}')
        with sp.Popen(args, stdout=sp.PIPE, text=True) as proc:
            model_count: int | None = None
            for line in proc.stdout:
                line = line.strip()
                log.debug(line)
                if line.startswith('c [appmc] Number of solutions is:'):
                    log.info(line)
                elif line.startswith('s mc '):
                    log.info(line)
                    assert model_count is None
                    model_count = int(line.removeprefix('s mc '))
                else:
                    log.debug(line)
        assert model_count is not None
        log.info(f'model count: 2^{log2(model_count):.1f} == {model_count}',
                 extra={'seed': seed, 'epsilon': epsilon, 'delta': delta, 'sampling_set': sampling_set})
        return model_count
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
    ddt: np.ndarray[Any, np.dtype[np.uint8]]
    block_size: int
    key_size: int
    sbox_bits: int
    num_rounds: int
    sbox_in: np.ndarray[Any, np.dtype[np.int32]]
    sbox_out: np.ndarray[Any, np.dtype[np.int32]]
    key: np.ndarray[Any, np.dtype[np.int32]]
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
        return list(model)
    def solve(self) -> Model:
        raw_model = self._solve()
        raw_model[0] = False
        raw_model = np.array(raw_model, dtype=np.uint8)
        model = self.get_model(raw_model)
        return model
    def count_probability_for_random_key(self, epsilon, delta, verbosity=2):
        assert self.key_size % 8 == 0
        key_bits = np.unpackbits(np.array(bytearray(os.urandom(self.key_size // 8))))
        key_bits = key_bits.reshape(self.key.shape)
        if self.key.shape[-1] == 4:
            key_nibbles = np.packbits(key_bits, axis=-1, bitorder='little')[..., 0]
            key_str = ''.join(f'{x:01x}' for x in key_nibbles)
        elif self.key.shape[-1] == 8:
            key_bytes = np.packbits(key_bits, axis=-1, bitorder='little')[..., 0]
            key_str = ''.join(f'{x:02x}' for x in key_bytes)
        else:
            raise ValueError('key must be composed of bytes or nibbles')
        sys.stdout.flush()
        cnf = copy.copy(self.cnf)
        cnf += CNF.create_xor(self.key.flatten(), rhs=key_bits.flatten())
        if verbosity == 0:
            print(f'key: {key_str}', end=' ', flush=True)
        num_solutions = count_solutions(cnf, epsilon, delta, verbosity=verbosity)
        if verbosity > 0:
            print(f'key: {key_str}', end=' ', flush=True)
        if num_solutions > 0:
            log2_prob = (log2(num_solutions)) - self.block_size
            print(f'probability: 2^{log2_prob:.2f}')
        else:
            print('probability: 0')
            return float('-inf')
        return key_bits, log2_prob
    def count_probability(self, epsilon: float, delta: float, verbosity: int=2):
        num_solutions = count_solutions(self.cnf, epsilon, delta, verbosity=verbosity)
        log2_prob = (log2(num_solutions)) - (self.block_size + self.key_size)
        print(f'probability: 2^{log2_prob:.2f}, {epsilon=}, {delta=}')
        return log2_prob
    def count_key_space(self, epsilon, delta, verbosity=2):
        """
        Use model counting to count the number of keys for which the
        characteristic is not impossible.
        """
        sampling_set = self.key.flatten().tolist()
        num_keys = count_solutions(self.cnf, epsilon, delta, verbosity=verbosity, sampling_set=sampling_set)
        log_num_keys = log2(num_keys)
        log.info(f'key space: 2^{log_num_keys:.2f}, {epsilon=}, {delta=}')
if __name__ == '__main__':
    from main import setup_logging
    setup_logging()
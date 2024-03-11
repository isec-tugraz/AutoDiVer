"""
Cipher model base classes
"""
from __future__ import annotations
from math import log2
import copy
from dataclasses import dataclass
import os
import logging
import random
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Any
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from sat_toolkit.formula import XorCNF, CNF, Truthtable
from pycryptosat import Solver
from .util import IndexSet, Model, fmt_log2
log = logging.getLogger(__name__)
@dataclass
class CountResult:
    probability: float
    key: None|np.ndarray[Any, np.dtype[np.uint8]] = None
    tweak: None|np.ndarray[Any, np.dtype[np.uint8]] = None
    pt: None|np.ndarray[Any, np.dtype[np.uint8]] = None
    def __repr__(self):
        fmt_array = lambda arr: np.packbits(arr, axis=-1, bitorder='little').tobytes().hex() if arr is not None else None
        return f'''CountResult(
    probability={self.probability},
    key={fmt_array(self.key)},
    tweak={fmt_array(self.tweak)},
    pt={fmt_array(self.pt)}
)'''
def count_solutions(cnf: XorCNF, epsilon: float, delta: float, verbosity: int=2, sampling_set: list[int] | None=None) -> int:
    sampling_set_log = f" over {len(sampling_set)} variables" if sampling_set is not None else ""
    log.info(f'counting solutions to cnf with {cnf.nvars} variables, {cnf.nclauses} clauses, and {cnf.nxor_clauses} xor clauses{sampling_set_log}, {epsilon=}, {delta=}')
    # create temporary file for cnf
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf') as f:
        if sampling_set is not None:
            sampling_set_str = ' '.join(str(x) for x in sampling_set) + ' 0'
            f.write(f'c ind {sampling_set_str}\n')
        # ApproxMC does not support XOR clauses, so we convert to a CNF
        f.write(cnf.to_cnf().to_dimacs())
        f.flush()
        # run approxmc
        seed = int.from_bytes(os.urandom(4), 'little')
        args = ['approxmc', f'--seed={seed}', f'-e{epsilon}', f'-d{delta}', '--sparse=1', f'-v{verbosity}', f.name]
        log.info(f'running: {" ".join(args)}')
        with sp.Popen(args, stdout=sp.PIPE, text=True) as proc:
            model_count: int | None = None
            for line in proc.stdout:
                line = line.strip()
                if not (line.startswith('c ') or line.startswith('s ')):
                    log.info(line)
                elif 'ERROR' in line:
                    line = line.removeprefix('c ').removeprefix('ERROR')
                    line = line.lstrip(': ')
                    log.error(f'[appmc] {line}')
                elif 'WARN' in line:
                    line = line.removeprefix('c ').removeprefix('WARNING').removeprefix('WARN')
                    line = line.lstrip(': ')
                    log.warning(f'[appmc] {line}')
                elif line.startswith('c Reduced to '):
                    log.info(line)
                elif line.startswith('c [appmc] Number of solutions is:'):
                    log.info(line)
                elif line.startswith('c [appmc+arjun] Total time'):
                    log.info(line)
                elif line.startswith('s mc '):
                    log.info(line)
                    assert model_count is None
                    model_count = int(line.removeprefix('s mc '))
                else:
                    log.debug(line)
        retcode = proc.wait()
        if retcode != 0:
            raise sp.CalledProcessError(retcode, args)
        assert model_count is not None
        log.info(f'model count: {fmt_log2(model_count)} == {model_count}',
                 extra={'seed': seed, 'epsilon': epsilon, 'delta': delta, 'sampling_set': sampling_set})
        return model_count
class DifferentialCharacteristic():
    num_rounds: int
    sbox_in: np.ndarray[Any, np.dtype[np.uint8]]
    sbox_out: np.ndarray[Any, np.dtype[np.uint8]]
    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        trail = []
        with open(characteristic_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                assert len(line) == 16
                line_deltas = [int(l, 16) for l in line[::-1]]
                trail.append(line_deltas)
        trail = np.array(trail)
        if len(trail) % 2 != 0:
            log.error(f'expected an even number of differences in {characteristic_path!r}')
            raise ValueError(f'expected an even number of differences in {characteristic_path!r}')
        sbox_in = trail[0::2]
        sbox_out = trail[1::2]
        return cls(sbox_in, sbox_out)
    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike):
        self.sbox_in = np.array(sbox_in, dtype=np.uint8)
        self.sbox_out = np.array(sbox_out, dtype=np.uint8)
        if self.sbox_in.shape != self.sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')
        self.num_rounds = len(self.sbox_in)
    def log2_ddt_probability(self, ddt: np.ndarray):
        ddt_prob = np.log2(ddt[self.sbox_in, self.sbox_out] / len(ddt)).sum()
        return ddt_prob
class SboxCipher(IndexSet):
    cipher_name: str
    sbox: np.ndarray[Any, np.dtype[np.uint8]]
    ddt: np.ndarray
    block_size: int
    key_size: int
    tweak_size: int = 0
    sbox_bits: int
    num_rounds: int
    sbox_in: np.ndarray[Any, np.dtype[np.int32]]
    sbox_out: np.ndarray[Any, np.dtype[np.int32]]
    key: np.ndarray[Any, np.dtype[np.int32]]
    pt: np.ndarray[Any, np.dtype[np.int32]]
    tweak: np.ndarray[Any, np.dtype[np.int32]]
    cnf: CNF
    _cnf_cache: dict[bytes, CNF] = {}
    def __init__(self, char: DifferentialCharacteristic):
        super().__init__()
        self.char = char
        self.num_rounds = char.num_rounds
        self.cnf = XorCNF()
    @classmethod
    def _get_cnf(cls, sbox, x_set):
        lut = np.zeros((len(sbox), len(sbox)), dtype=np.uint8)
        lut[x_set, sbox[x_set]] = 1
        assert lut.sum() == len(x_set)
        dnf = Truthtable.from_lut(lut.T.flatten())
        cache_key = dnf.on_set.astype(np.int32).tobytes()
        try:
            # return a copy from the cache
            return CNF(cls._cnf_cache[cache_key])
        except KeyError:
            pass
        cnf = dnf.to_cnf()
        cls._cnf_cache[cache_key] = cnf
        # don't return the original, because it's mutable
        return CNF(cnf)
    def get_solution_set_cnf(self, delta_in, delta_out):
        x = np.arange(len(self.sbox), dtype=np.uint8)
        x_set, = np.where(self.sbox[x] ^ self.sbox[x ^ delta_in] == delta_out)
        return self._get_cnf(self.sbox, x_set)
    def _model_sboxes(self, sbox_in: None|np.ndarray[Any, np.dtype[np.int32]]=None, sbox_out: None|np.ndarray[Any, np.dtype[np.int32]]=None):
        sbox_in = sbox_in if sbox_in is not None else self.sbox_in
        sbox_out = sbox_out if sbox_out is not None else self.sbox_out
        inp = sbox_in.reshape(-1, self.sbox_bits)
        out = sbox_out.reshape(-1, self.sbox_bits)
        delta_in = self.char.sbox_in.reshape(-1)
        delta_out = self.char.sbox_out.reshape(-1)
        # print(f'{inp.shape = }', f'{out.shape}')
        # print(f'{delta_in.shape = }', f'{delta_out.shape}')
        self._actual_sbox_in = inp.copy()
        self._actual_sbox_out = out.copy()
        self._fieldnames.add('_actual_sbox_in')
        self._fieldnames.add('_actual_sbox_out')
        sbox_cnf = CNF()
        for inp, out, delta_in, delta_out in zip(inp, out, delta_in, delta_out):
            mapping = np.concatenate((np.array([0], dtype=np.int32), inp, out))
            cnf = self.get_solution_set_cnf(delta_in, delta_out).translate(mapping)
            sbox_cnf += cnf
        self.cnf += sbox_cnf
    def _model_linear_layer(self):
        raise NotImplementedError("this should be implemented by subclasses")
    def _solve(self):
        seed = int.from_bytes(os.urandom(4), 'little')
        args = ['cryptominisat5', f'--random={seed}', '--polar=rnd']
        log.info(args)
        log.info(f'solving with {args} #Clauses: {len(self.cnf._clauses)}, #XORs: {len(self.cnf._xor_clauses)}, #Vars: {self.cnf.nvars}')
        is_sat, model = self.cnf.solve_dimacs(args)
        if not is_sat:
            log.info('RESULT cnf is UNSAT')
            raise ValueError('cnf is UNSAT')
        log.info('RESULT cnf is SAT')
        return list(model)
    @classmethod
    def _fmt_arr(cls, arr: np.ndarray, cellsize: int):
        if cellsize == 0 and len(arr) == 0:
            return ''
        if cellsize == 4:
            return ''.join(f'{x:01x}' for x in arr.flatten())
        if cellsize == 8:
            return ''.join(f'{x:02x}' for x in arr.flatten())
        raise ValueError(f'cellsize must be 4 or 8 not {cellsize}')
    def solve(self) -> Model:
        raw_model = self._solve()
        raw_model[0] = False
        raw_model = np.array(raw_model, dtype=np.uint8)
        model = self.get_model(raw_model)
        key_str = self._fmt_arr(model.key, self.key.shape[-1]) # type: ignore
        tweak_str = self._fmt_arr(model.tweak, self.tweak.shape[-1]) # type: ignore
        pt_str = self._fmt_arr(model.pt, self.pt.shape[-1]) # type: ignore
        log.info(f'RESULT key={key_str}, tweak={tweak_str}, pt={pt_str}')
        return model
    def _fmt_tweak_or_key(self, key_bits: np.ndarray):
        if self.key.shape[-1] == 4:
            key_nibbles = np.packbits(key_bits, axis=-1, bitorder='little')[..., 0]
            key_str = ''.join(f'{x:01x}' for x in key_nibbles)
        elif self.key.shape[-1] == 8:
            key_bytes = np.packbits(key_bits, axis=-1, bitorder='little')[..., 0]
            key_str = ''.join(f'{x:02x}' for x in key_bytes)
        else:
            raise ValueError('key must be composed of bytes or nibbles')
        return key_str
    def _get_random_pt(self):
        return np.unpackbits(np.array(bytearray(os.urandom(self.block_size // 8))))
    def count_probability(self, epsilon: float, delta: float, fixed_key: bool=False, fixed_tweak: bool=False, fixed_pt: bool=False,verbosity: int=2) -> CountResult:
        assert self.key_size % 8 == 0
        assert self.tweak_size % 8 == 0
        assert self.block_size % 8 == 0
        assert np.prod(self.key.shape) == self.key_size #type: ignore
        assert np.prod(self.tweak.shape) == self.tweak_size #type: ignore
        assert np.prod(self.pt.shape) == self.block_size #type: ignore
        denominator_log2 = self.block_size + self.tweak_size + self.key_size
        constraints_description = []
        key_bits = None
        tweak_bits = None
        cnf = copy.copy(self.cnf)
        if fixed_key:
            key_bits = np.unpackbits(np.array(bytearray(os.urandom(self.key_size // 8))))
            key_bits = key_bits.reshape(-1, self.key.shape[-1])
            key_str = self._fmt_tweak_or_key(key_bits)
            constraints_description.append(f'key={key_str}')
            cnf += CNF.create_xor(self.key.flatten(), rhs=key_bits.flatten())
            denominator_log2 -= self.key_size
        if fixed_tweak:
            tweak_bits = np.unpackbits(np.array(bytearray(os.urandom(self.tweak_size // 8))))
            tweak_bits = tweak_bits.reshape(-1, self.tweak.shape[-1])
            tweak_str = self._fmt_tweak_or_key(tweak_bits)
            constraints_description.append(f'tweak={tweak_str}')
            cnf += CNF.create_xor(self.tweak.flatten(), rhs=tweak_bits.flatten())
            denominator_log2 -= self.tweak_size
        if fixed_pt:
            pt_bits = self._get_random_pt()
            pt_bits = pt_bits.reshape(-1, self.pt.shape[-1])
            pt_str = self._fmt_tweak_or_key(pt_bits)
            constraints_description.append(f'pt={pt_str}')
            cnf += CNF.create_xor(self.pt.flatten(), rhs=pt_bits.flatten())
            denominator_log2 -= self.block_size
        log_str = 'probability' if not constraints_description else f'probability for {", ".join(constraints_description)}'
        log.info(f'counting {log_str}')
        num_solutions = count_solutions(cnf, epsilon, delta, verbosity=verbosity)
        prob = num_solutions / 2**denominator_log2
        log.info(f'RESULT {log_str}: {fmt_log2(prob)}')
        return CountResult(prob, key_bits, tweak_bits)
    def count_tweakey_space(self, epsilon, delta, count_key: bool=True, count_tweak: bool=True, verbosity: int=2):
        """
        Use model counting to count the number of tweakeys for which the
        characteristic is not impossible.
        """
        sampling_set = []
        if count_key:
            sampling_set += self.key.flatten().tolist()
        if count_tweak:
            sampling_set += self.tweak.flatten().tolist()
        name = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]
        num_keys = count_solutions(self.cnf, epsilon, delta, verbosity=verbosity, sampling_set=sampling_set)
        log_num_keys = log2(num_keys)
        log.info(f'RESULT {name} space: 2^{log_num_keys:.2f}, {epsilon=}, {delta=}')
    def count_tweakey_space_sat_solver(self, trials: int, count_key: bool=True, count_tweak: bool=True, verbosity: int=2):
        """
        Use repeated SAT solving to estimate the number of tweakeys for which
        the characteristic is not impossible.
        """
        sampling_set = []
        if count_key:
            sampling_set += self.key.flatten().tolist()
        if count_tweak:
            sampling_set += self.tweak.flatten().tolist()
        name = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]
        solver = Solver()
        solver.add_clauses(self.cnf._clauses)
        for xor_clause in self.cnf._xor_clauses:
            rhs = int(np.prod(np.sign(xor_clause))) < 0
            pos_clause = np.abs(xor_clause).tolist()
            solver.add_xor_clause(pos_clause, rhs=rhs)
        log.info(f'solving with CryptoMiniSat #Clauses: {len(self.cnf._clauses)}, #XORs: {len(self.cnf._xor_clauses)}, #Vars: {self.cnf.nvars}')
        count_sat = 0
        pbar = tqdm(range(trials))
        for i in pbar:
            sampling_new = [x * (-1)**random.randint(0, 1) for x in sampling_set]
            is_sat, _ = solver.solve(sampling_new)
            count_sat += is_sat
            pbar.set_description(f'valid {name}s: {count_sat}/{i + 1}')
        log.info(f'RESULT {name} count: {count_sat}/{trials}')
        # num_keys = count_sat
        # log_num_keys = log2(num_keys)
        # log.info(f'RESULT {name} space: 2^{log_num_keys:.2f}, {epsilon=}, {delta=}')
        key_cnf = CNF([], nvars=self.cnf.nvars)
        try:
            pbar = tqdm()
            solver.start_getting_small_clauses(1<<32 - 1, 1<<32 - 1)
            min_var = min(sampling_set)
            max_var = max(sampling_set)
            sampling_set = set(sampling_set)
            while True:
                clause = solver.get_next_small_clause()
                if clause is None:
                    break
                clause = np.array(clause)
                variables = np.abs(clause)
                if np.all((variables >= min_var) & (variables <= max_var)):
                    if any(variable not in sampling_set for variable in variables):
                        continue
                    tqdm.write(self.format_clause(clause))
                    key_cnf.add_clause(clause)
                pbar.update()
        finally:
            pbar.close()
            solver.end_getting_small_clauses()
        min_key_cnf = key_cnf.minimize_espresso()
        log.info(f'key conditions: {min_key_cnf!r}')
        for clause in min_key_cnf:
            log.info(self.format_clause(np.array(clause)))
if __name__ == '__main__':
    from main import setup_logging
    setup_logging()
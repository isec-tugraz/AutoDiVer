"""
Cipher model base classes
"""
from __future__ import annotations

from math import log2
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import logging
import random
import subprocess as sp
import tempfile
import time
from pathlib import Path
from typing import Any, Literal
from itertools import count
import re
import shutil
import sys

from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from galois import GF2
from sat_toolkit.formula import XorCNF, CNF, XorClauseList, Truthtable, Clause
from pycryptosat import Solver
from autodiver import version
from .util import IndexSet, Model, fmt_log2


log = logging.getLogger(__name__)

class UnsatException(Exception):
    pass

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

def _available_cpus():
    try:
        # only available on Linux
        return len(os.sched_getaffinity(0)) # type: ignore
    except AttributeError:
        return os.cpu_count()


@dataclass
class _ApproxMcLoggingContext:
    pbar: tqdm|None=None
    threshold: int|None=None
    round_iter: int|None=None
    num_hashes: int|None=None
    model_count: int|None=None

    def processs_log_line(self, line: str):
        line = line.strip()
        if not (line.startswith('c ') or line.startswith('s ')) and line != 'c':
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

        elif line.startswith('c [appmc] threshold set to'):
            log.info(line)
            self.threshold = int(line.removeprefix('c [appmc] threshold set to ').split()[0])

        elif 'round: ' in line and 'hashes: ' in line:
            match_round = re.search(r'round:\s+(\d+)', line)
            match_hashes = re.search(r'hashes:\s+(\d+)', line)
            if match_round and match_hashes:
                round_iter = int(match_round.group(1))
                num_hashes = int(match_hashes.group(1))

                self.pbar = tqdm(total=self.threshold, desc=f'round {round_iter:2d}, hashes: {num_hashes:3d}')


        elif line.startswith('c [appmc] bounded_sol_count'):
            match_sol = re.search(r'ret:\s+l_(True|False)', line)
            match_sol_number = re.search(r'sol no.\s+(\d+)', line)

            if match_sol:
                sol_found = match_sol.group(1).lower() == 'true'

                if not sol_found and self.pbar is not None:
                    self.pbar.close()
                    self.pbar = None

                if match_sol_number:
                    sol_number = int(match_sol_number.group(1))
                    if self.pbar is not None:
                        self.pbar.update(sol_number - self.pbar.n)
                        if not sol_found or sol_number == self.threshold:
                            self.pbar.close()
                            self.pbar = None

        elif line.startswith('c [appmc+arjun] Total time'):
            log.info(line)

        elif line.startswith('s mc '):
            log.info(line)

            assert self.model_count is None
            self.model_count = int(line.removeprefix('s mc '))
        else:
            pass

        if self.pbar:
            self.pbar.refresh()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pbar is not None:
            self.pbar.__exit__(exc_type, exc_value, traceback)


def count_solutions(cnf: XorCNF, epsilon: float, delta: float, verbosity: int=2, sampling_set: list[int] | None=None, seed: int|None=None) -> int:
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
        if seed is None:
            seed = int.from_bytes(os.urandom(4), 'little')
        args = ['approxmc', f'--seed={seed}', f'--{epsilon=}', f'--{delta=}', '--sparse=1', f'--verb={verbosity}', f.name]

        log.info(f'running: {" ".join(args)}')
        with sp.Popen(args, stdout=sp.PIPE, text=True) as proc:
            assert proc.stdout is not None

            with _ApproxMcLoggingContext() as ctx:
                for line in proc.stdout:
                    ctx.processs_log_line(line)

            model_count = ctx.model_count
            assert model_count is not None

        retcode = proc.wait()
        if retcode != 0:
            raise sp.CalledProcessError(retcode, args)

        log.info(f'model count: {fmt_log2(model_count)} == {model_count}',
                 extra={'seed': seed, 'epsilon': epsilon, 'delta': delta, 'sampling_set': sampling_set})
        return model_count


class DifferentialCharacteristic():
    num_rounds: int
    sbox_in: np.ndarray[Any, np.dtype[np.uint8]]
    sbox_out: np.ndarray[Any, np.dtype[np.uint8]]

    file_path: Path|None

    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        trail = []
        with open(characteristic_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                assert len(line) == 16 or len(line) == 32
                line_deltas = [int(l, 16) for l in line[::-1]]
                trail.append(line_deltas)

        trail = np.array(trail)
        if len(trail) % 2 != 0:
            log.error(f'expected an even number of differences in {characteristic_path!r}')
            raise ValueError(f'expected an even number of differences in {characteristic_path!r}')

        sbox_in = trail[0::2]
        sbox_out = trail[1::2]

        return cls(sbox_in, sbox_out, file_path=characteristic_path)

    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike, file_path: Path|None=None):
        self.sbox_in = np.array(sbox_in, dtype=np.uint8)
        self.sbox_out = np.array(sbox_out, dtype=np.uint8)
        self.file_path = file_path
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
    sbox_assumptions: np.ndarray[Any, np.dtype[np.int32]]
    cnf: CNF

    model_type: Literal['solution_set', 'split_solution_set']
    _cnf_cache: dict[bytes, CNF] = {}

    def __init__(self, char: DifferentialCharacteristic, *, model_type: Literal['solution_set', 'split_solution_set'] = 'solution_set', model_sbox_assumptions: bool = False):
        super().__init__()

        if model_type not in ('solution_set', 'split_solution_set'):
            raise ValueError(f'unknown model_type {model_type}')

        self.char = char
        self.num_rounds = char.num_rounds
        self.cnf = XorCNF()

        self.model_type = model_type
        self.model_sbox_assumptions = model_sbox_assumptions
        self.__model_sboxes_called = False

    @classmethod
    def _lut_to_cnf(cls, lut: np.ndarray) -> CNF:
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

    @classmethod
    def _get_cnf(cls, sbox, x_set, *, model_type: Literal['solution_set', 'split_solution_set']):
        lut = np.zeros((len(sbox), len(sbox)), dtype=np.uint8)

        if model_type == 'solution_set':
            lut[x_set, sbox[x_set]] = 1
            assert lut.sum() == len(x_set)
        elif model_type == 'split_solution_set':
            lut[x_set, :] = 1
            lut[:, sbox[x_set]] = 1
        else:
            raise ValueError(f'unknown model_type {model_type}')

        return cls._lut_to_cnf(lut)



    def log_result(self, **kwargs):
        # log results in a machine readable format
        # gather results

        git_cmd = shutil.which('git')
        git_commit = git_cmd and sp.check_output([git_cmd, 'rev-parse', 'HEAD']).decode().strip()
        git_changed_files = git_cmd and sp.check_output([git_cmd, 'status', '--porcelain', '-uno', '-z']).decode().strip('\0').split('\0')
        hostname = sp.check_output(['hostname']).decode().strip()
        available_cpus = _available_cpus()

        context = {
            'cipher': type(self).__name__,
            'model_type': self.model_type,
            'hostname': hostname,
            'available_cpus': available_cpus,
            'char': {
                'num_rounds': self.char.num_rounds,
                'file_path': self.char.file_path,
                'type': type(self.char).__name__,
            },
            'argv': sys.argv,
            'git': {
                'commit': git_commit,
                'changed_files': git_changed_files,
            },
            'version': version,
        }

        extra = {
            'context': context,
            **kwargs,
        }

        log.debug(f'RESULT', extra=extra)

    def _get_sbox_cnf(self, delta_in, delta_out):
        x = np.arange(len(self.sbox), dtype=np.uint8)
        x_set, = np.where(self.sbox[x] ^ self.sbox[x ^ delta_in] == delta_out)
        return self._get_cnf(self.sbox, x_set, model_type=self.model_type)

    def _model_sboxes(self, sbox_in: None|np.ndarray[Any, np.dtype[np.int32]]=None, sbox_out: None|np.ndarray[Any, np.dtype[np.int32]]=None):
        if self.__model_sboxes_called:
            raise ValueError('model_sboxes can only be called once')
        self.__model_sboxes_called = True

        sbox_in = sbox_in if sbox_in is not None else self.sbox_in
        sbox_out = sbox_out if sbox_out is not None else self.sbox_out

        assert sbox_in.shape[1:] == sbox_out.shape[1:]

        # support using sbox_in[r+1] as ciphertext variables
        assert sbox_in.shape[0] == sbox_out.shape[0] or self.sbox_in.shape[0] == self.sbox_out.shape[0] + 1

        inp_vars = sbox_in.reshape(-1, self.sbox_bits)
        out_vars = sbox_out.reshape(-1, self.sbox_bits)

        delta_in = self.char.sbox_in.reshape(-1)
        delta_out = self.char.sbox_out.reshape(-1)

        self._actual_sbox_in = inp_vars.copy()
        self._actual_sbox_out = out_vars.copy()

        self._fieldnames.add('_actual_sbox_in')
        self._fieldnames.add('_actual_sbox_out')

        if self.model_sbox_assumptions:
            self.add_index_array("sbox_assumptions", sbox_out.shape[:-1])
        else:
            self.add_index_array("sbox_assumptions", (0,))

        sbox_cnf = CNF()
        # print(f'inp_vars: {sbox_in.shape}, out_vars: {sbox_out.shape}, delta_in: {self.char.sbox_in.shape}, delta_out: {self.char.sbox_out.shape}, assumptions: {self.sbox_assumptions.shape}')

        for idx in range(delta_out.shape[0]):
            inp, out = inp_vars[idx], out_vars[idx]
            d_in, d_out = delta_in[idx], delta_out[idx]

            mapping = np.concatenate((np.array([0], dtype=np.int32), inp, out))
            cnf = self._get_sbox_cnf(d_in, d_out).translate(mapping)

            if self.model_sbox_assumptions:
                assumption = self.sbox_assumptions.ravel()[idx]
                cnf = cnf.implied_by(assumption)

            sbox_cnf += cnf

        self.cnf += sbox_cnf


    def _model_linear_layer(self):
        raise NotImplementedError("this should be implemented by subclasses")

    def _solve(self, cnf: CNF=None, *, log_result: bool=True, seed: int|None=None, assumptions: np.ndarray|None=None):
        if assumptions is None:
            assumptions = self.sbox_assumptions

        seed = int.from_bytes(os.urandom(4), 'little') if seed is None else seed
        args = ['cryptominisat5', f'--random={seed}', '--polar=rnd']

        cnf = cnf if cnf is not None else self.cnf
        with tempfile.NamedTemporaryFile(mode='w', prefix='assumptions', suffix='.txt') as f:
            f.write("\n".join(str(x) for x in assumptions.ravel()))
            f.flush()

            args += ['--assump', f.name]

            if log_result:
                log.info(f'solving with {args} #Clauses: {len(cnf._clauses)}, #XORs: {len(cnf._xor_clauses)}, #Vars: {cnf.nvars}, #Assumptions: {assumptions.size}')
            is_sat, model = cnf.solve_dimacs(args)

        if not is_sat:
            if log_result:
                log.info('RESULT cnf is UNSAT')
            raise UnsatException('cnf is UNSAT')

        if log_result:
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


    def solve(self, seed: int|None=None) -> Model:
        start_time = time.monotonic()
        if seed is None:
            seed = int.from_bytes(os.urandom(4), 'little')
        try:
            raw_model = self._solve(seed=seed)
        except UnsatException:
            self.log_result(solve_result={'status': 'UNSAT'})
            raise

        end_time = time.monotonic()

        raw_model[0] = False
        raw_model = np.array(raw_model, dtype=np.uint8)

        model = self.get_model(raw_model)

        key_str = self._fmt_arr(model.key, self.key.shape[-1]) # type: ignore
        tweak_str = self._fmt_arr(model.tweak, self.tweak.shape[-1]) # type: ignore
        pt_str = self._fmt_arr(model.pt, self.pt.shape[-1]) # type: ignore

        solve_result = {
            'status': 'SAT',
            'key': key_str,
            'tweak': tweak_str,
            'pt': pt_str,
            'time': end_time - start_time,
            'seed': seed,
        }
        self.log_result(solve_result=solve_result)

        log.info(f'RESULT key={key_str}, tweak={tweak_str}, pt={pt_str}')

        return model

    def find_conflict(self) -> np.ndarray[Any, np.dtype[np.int32]]:
        if not self.model_sbox_assumptions:
            raise ValueError('model_sbox_assumptions must be True to find conflicts')

        solver = Solver()
        solver.add_clauses(self.cnf._clauses)
        for xor_clause in self.cnf._xor_clauses:
            rhs = int(np.prod(np.sign(xor_clause))) > 0
            pos_clause = np.abs(xor_clause).tolist()
            solver.add_xor_clause(pos_clause, rhs=rhs)

        assumptions_list = self.sbox_assumptions.ravel().tolist()
        log.info(f'solving CNF with #Clauses: {len(self.cnf._clauses)}, #XORs: {len(self.cnf._xor_clauses)}, #Vars: {self.cnf.nvars}, #Assumptions: {len(assumptions_list)}')

        start_time = time.monotonic()
        is_sat, raw_model = solver.solve(assumptions_list)
        end_time = time.monotonic()
        if is_sat:
            log.info('RESULT cnf is satisfiable, no conflict')

            raw_model = list(raw_model)
            raw_model[0] = False
            raw_model = np.array(raw_model, dtype=np.uint8)
            model = self.get_model(raw_model)
            key_str = self._fmt_arr(model.key, self.key.shape[-1]) # type: ignore
            tweak_str = self._fmt_arr(model.tweak, self.tweak.shape[-1]) # type: ignore
            pt_str = self._fmt_arr(model.pt, self.pt.shape[-1]) # type: ignore
            solve_result = {
                'status': 'SAT',
                'key': key_str,
                'tweak': tweak_str,
                'pt': pt_str,
                'time': end_time - start_time,
            }
            self.log_result(solve_result=solve_result)
            log.info(f'RESULT key={key_str}, tweak={tweak_str}, pt={pt_str}')

            return np.array([], dtype=np.int32)

        conflict = np.array(solver.get_conflict(), dtype=np.int32)
        log.info(f'RESULT conflict: {self.format_clause(conflict)}')
        return conflict

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

        key_str = ""
        tweak_str = ""
        pt_str = ""

        cnf = copy.copy(self.cnf)
        if fixed_key:
            key_bits = np.unpackbits(np.array(bytearray(os.urandom(self.key_size // 8))))
            key_bits = key_bits.reshape(-1, self.key.shape[-1])
            key_str = self._fmt_tweak_or_key(key_bits)
            key_str = self._fmt_arr(model.key, self.key.shape[-1]) # type: ignore
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


        # while providing a sampling set should speed things up in theory
        # approxmc is faster if we don't provide a sampling set
        # (maybe Arjun is better at simplifying if we don't provide a sampling set?)
        # nr_half = len(self.sbox_in) // 2
        # sampling_set = (self.key.flatten().tolist()
        #                 + self.tweak.flatten().tolist()
        #                 + self.sbox_in[nr_half].flatten().tolist()
        #                 + self.sbox_out[nr_half].flatten().tolist())
        sampling_set = None

        start_time = time.monotonic()
        seed = int.from_bytes(os.urandom(4), 'little')
        num_solutions = count_solutions(cnf, epsilon, delta, verbosity=verbosity, sampling_set=sampling_set, seed=seed)
        end_time = time.monotonic()

        # num_inactive_sboxes = int((self.char.sbox_in == 0).sum())
        # denominator_log2 += num_inactive_sboxes * self.sbox_bits

        prob = num_solutions / (1<<denominator_log2)

        log.info(f'RESULT {log_str}: {fmt_log2(prob)}')

        count_result = {
            'probability': prob,
            'epsilon': epsilon,
            'delta': delta,
            'key' : key_str,
            'tweak': tweak_str,
            'pt': pt_str,
            'time': end_time - start_time,
            'seed': seed,
        }
        self.log_result(count_result=count_result)

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

        start_time = time.monotonic()
        seed = int.from_bytes(os.urandom(4), 'little')
        num_keys = count_solutions(self.cnf, epsilon, delta, verbosity=verbosity, sampling_set=sampling_set, seed=seed)
        end_time = time.monotonic()

        log_num_keys = fmt_log2(num_keys)
        log.info(f'RESULT {name} space: {log_num_keys}, {epsilon=}, {delta=}')

        count_tweakey_result = {
            'num_keys': num_keys,
            'epsilon': epsilon,
            'delta': delta,
            'count_key': count_key,
            'count_tweak': count_tweak,
            'time': end_time - start_time,
            'seed': seed,
        }
        self.log_result(count_tweakey_result=count_tweakey_result)


    @staticmethod
    def get_small_clauses(solver: Solver, max_len: int, max_glue: int):
        try:
            solver.start_getting_small_clauses(max_len, max_glue)
            while True:
                clause = solver.get_next_small_clause()
                if clause is None:
                    break
                yield clause
        finally:
            solver.end_getting_small_clauses()

    @staticmethod
    def get_small_clauses_over_set(solver: Solver, max_len: int, max_glue: int, sampling_set: set[int]):
        min_var = min(sampling_set)
        max_var = max(sampling_set)

        for clause in SboxCipher.get_small_clauses(solver, max_len, max_glue):
            clause = sorted(clause, key=abs)
            np_clause = np.array(clause)
            variables = np.abs(np_clause)

            if np.any((variables < min_var) | (variables > max_var)):
                continue
            if any(variable not in sampling_set for variable in variables):
                continue

            yield clause

    def count_lin_tweakey_space(self, count_key: bool=True, count_tweak: bool=True):
        """
        count the size of the tweakey space under the assumption that it is a vector space
        """
        sampling_set_list = []
        if count_key:
            sampling_set_list += self.key.flatten().tolist()
        if count_tweak:
            sampling_set_list += self.tweak.flatten().tolist()

        sampling_set_list = np.array(sampling_set_list, dtype=np.int32)
        sampling_set = set(sampling_set_list)

        # one extra sample because the affine space has an offset as well
        count_initial_samples = len(sampling_set) + 1
        initial_samples = []

        start_time = time.monotonic()
        name = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]
        with ThreadPoolExecutor(max_workers=_available_cpus()) as executor:
            def task(_index):
                raw_model = self._solve(log_result=False)
                raw_model[0] = False
                raw_model = np.array(raw_model, dtype=np.uint8)
                return raw_model[sampling_set_list]

            samples = executor.map(task, range(count_initial_samples))
            for sample in tqdm(samples, total=count_initial_samples, desc=f'gathering valid {name}s'):
                initial_samples.append(sample)

        counter_example_found = True
        while counter_example_found:
            samples = GF2(initial_samples)

            # describe the affine space spanned by the samples
            const = samples[0].copy()
            samples += const
            lin_space = samples.row_space()

            # heuristically search for constant offset with lower hamming weight
            for vec in lin_space:
                new_const = const + vec
                if np.array(new_const, int).sum() < np.array(const, int).sum():
                    const = new_const

            log.info(f'gathered {name}s span affine space of dimension {len(lin_space)}')

            # we now have an affine space for the possible keys:
            # K = const + v * lin_space (for all v)
            # we can multiply with right_kern, the right kernel of lin_space
            # K * right_kern = const * right_kern
            # right_kern.T * K = right_kern.T * const

            right_kern = lin_space.T.left_null_space().T
            assert np.all(lin_space @ right_kern == 0)

            A = right_kern.T
            b = right_kern.T @ const

            extra_vars = range(self.cnf.nvars + 1, self.cnf.nvars + len(A) + 1)
            extra_constraints = XorCNF()
            extra_constraints.nvars = self.cnf.nvars + len(A)

            for i, eq, rhs in zip(count(), A, b):
                variables, = np.nonzero(eq)
                var_idxes = sampling_set_list[variables]
                var_names = self.describe_idx_array(var_idxes)

                # create xors with extra variables that track whether the xor is satisfied
                extra_constraints += XorCNF.create_xor(*var_idxes[:, np.newaxis], [extra_vars[i]], rhs=[rhs])

            # at least one of the xors must be violated
            extra_constraints.add_clauses(list(extra_vars) + [0])
            try:
                log.info('solving for counterexample')
                raw_model = self._solve(self.cnf + extra_constraints, log_result=False)
                log.info('found counterexample -> trying again')
                raw_model[0] = False
                raw_model = np.array(raw_model, dtype=np.uint8)
                sample = raw_model[sampling_set_list]
                initial_samples.append(sample)
                counter_example_found = True
            except UnsatException as e:
                assert 'UNSAT' in str(e)
                log.info(f'RESULT no counterexample found -> conditions on {name} are necessary')
                counter_example_found = False
                break
        end_time = time.monotonic()
        assert A is not None and b is not None

        constraints = []
        for i, eq, rhs in zip(count(), A, b):
            variables, = np.nonzero(eq)
            var_idxes = sampling_set_list[variables]
            var_names = self.describe_idx_array(var_idxes)

            str_lhs = ' ⊕ '.join(var_names)
            constr = f'{str_lhs} = {rhs}'
            constraints.append(constr)
            log.info(f'RESULT {constr}')

        count_tweakey_lin_result = {
            'constraints': constraints,
            'count_key': count_key,
            'count_tweak': count_tweak,
            'time': end_time - start_time,
        }

        self.log_result(count_tweakey_lin_result=count_tweakey_lin_result)



    def count_tweakey_space_sat_solver(self, trials: int, count_key: bool=True, count_tweak: bool=True, verbosity: int=2):
        """
        Use repeated SAT solving to estimate the number of tweakeys for which
        the characteristic is not impossible.
        """
        sampling_set_list = []
        if count_key:
            sampling_set_list += self.key.flatten().tolist()
        if count_tweak:
            sampling_set_list += self.tweak.flatten().tolist()
        sampling_set = set(sampling_set_list)

        name = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]

        solver = Solver()
        solver.add_clauses(self.cnf._clauses)
        for xor_clause in self.cnf._xor_clauses:
            rhs = int(np.prod(np.sign(xor_clause))) > 0
            pos_clause = np.abs(xor_clause).tolist()
            solver.add_xor_clause(pos_clause, rhs=rhs)

        log.info(f'solving with CryptoMiniSat #Clauses: {len(self.cnf._clauses)}, #XORs: {len(self.cnf._xor_clauses)}, #Vars: {self.cnf.nvars}')

        count_sat = 0
        count_unsat = 0

        key_cnf = CNF([], nvars=self.cnf.nvars)

        prev_small_clauses_check = time.monotonic()
        prev_keys_update = 0

        start_time = time.monotonic()
        pbar = tqdm(range(trials))
        for i in pbar:
            sampling_new = [x * (-1)**random.randint(0, 1) for x in sampling_set_list]
            is_sat, _ = solver.solve(self.sbox_assumptions.ravel().tolist() + sampling_new)
            count_sat += is_sat
            count_unsat += not is_sat

            if time.monotonic() - prev_keys_update >= 0.1:
                prev_keys_update = time.monotonic()
                pbar.set_description(f'valid {name}s: {count_sat}/{i + 1}')

            # check for learnt clauses over the sampling set if we haven't done
            # so in the last 10 seconds
            if time.monotonic() - prev_small_clauses_check <= 10:
                continue
            prev_small_clauses_check = time.monotonic()

            for clause in self.get_small_clauses_over_set(solver, len(sampling_set), 1<<32 - 1, sampling_set):
                clause = Clause(clause)
                if clause not in key_cnf:
                    tqdm.write(self.format_clause(clause))
                    key_cnf.add_clause(clause)
        end_time = time.monotonic()

        log.info(f'RESULT {name} count: {count_sat}/{trials}')

        for clause in self.get_small_clauses_over_set(solver, len(sampling_set), 1<<32 - 1, sampling_set):
            clause = Clause(clause)

            if clause not in key_cnf:
                tqdm.write(self.format_clause(clause))
                key_cnf.add_clause(clause)

        min_key_cnf = key_cnf.minimize_espresso()
        log.info(f'key conditions: {min_key_cnf!r}')

        key_conditions = []
        for clause in min_key_cnf:
            formatted = self.format_clause(np.array(clause))
            key_conditions.append(formatted)
            log.info(formatted)

        count_tweakeys_sat_result = {
            'count_sat': count_sat,
            'count_unsat': count_unsat,
            'trials': trials,
            'count_key': count_key,
            'count_tweak': count_tweak,
            'tweakey_conditions': key_conditions,
            'time': end_time - start_time,
        }

        self.log_result(count_tweakeys_sat_result=count_tweakeys_sat_result)

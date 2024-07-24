"""
Cipher model base classes
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import logging
import random
import subprocess as sp
import tempfile
import time
from typing import Any, Literal
from itertools import count
import shutil
import sys

from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from galois import GF2
from sat_toolkit.formula import XorCNF, CNF, Clause
from pycryptosat import Solver

from . import version
from .util import IndexSet, Model, fmt_log2
from .gf2_util import affine_hull
from .sat_util import UnsatException, count_solutions, lut_to_cnf, xor_cnf_as_cryptominisat_solver
from .characteristic import DifferentialCharacteristic


log = logging.getLogger(__name__)


class Timer:
    start: float
    end: float
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()

    def __str__(self) -> str:
        return f'{self.end - self.start:.3f} seconds'

    def elapsed(self) -> float:
        return self.end - self.start

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

    def log_result(self, **kwargs):
        """log results in machine readable json"""
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

    @staticmethod
    def _get_cnf(sbox: np.ndarray, x_set: np.ndarray, *, model_type: Literal['solution_set', 'split_solution_set']) -> CNF:
        lut = np.zeros((len(sbox), len(sbox)), dtype=np.uint8)

        if model_type == 'solution_set':
            lut[x_set, sbox[x_set]] = 1
            assert lut.sum() == len(x_set)
        elif model_type == 'split_solution_set':
            lut[x_set, :] = 1
            lut[:, sbox[x_set]] = 1
        else:
            raise ValueError(f'unknown model_type {model_type}')

        return lut_to_cnf(lut)

    def _get_sbox_cnf(self, delta_in, delta_out) -> CNF:
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

    def _solve(self, cnf: CNF=None, *, log_result: bool=True, seed: int|None=None, assumptions: np.ndarray|None=None) -> np.ndarray[Any, np.dtype[np.uint8]]:
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

        result = np.zeros(len(model), dtype=np.uint8)
        result[1:] = model[1:] # model[0] is always None
        return result

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
        with Timer() as timer:
            if seed is None:
                seed = int.from_bytes(os.urandom(4), 'little')
            try:
                raw_model = self._solve(seed=seed)
            except UnsatException:
                self.log_result(solve_result={'status': 'UNSAT'})
                raise

        model = self.get_model(raw_model)

        key_str = self._fmt_arr(model.key, self.key.shape[-1]) # type: ignore
        tweak_str = self._fmt_arr(model.tweak, self.tweak.shape[-1]) # type: ignore
        pt_str = self._fmt_arr(model.pt, self.pt.shape[-1]) # type: ignore

        solve_result = {
            'status': 'SAT',
            'key': key_str,
            'tweak': tweak_str,
            'pt': pt_str,
            'time': timer.elapsed(),
            'seed': seed,
        }
        self.log_result(solve_result=solve_result)

        log.info(f'RESULT key={key_str}, tweak={tweak_str}, pt={pt_str}')

        return model

    def find_conflicts(self) -> CNF:
        if not self.model_sbox_assumptions:
            raise ValueError('model_sbox_assumptions must be True to find conflicts')

        assumptions_list = self.sbox_assumptions.ravel().tolist()

        with Timer() as timer:
            conflicts = self._find_conflics(self.cnf, assumptions_list)

        formatted_conflicts = [self.format_clause(clause) for clause in conflicts]

        log.info(f'RESULT conflict: {conflicts!r}')
        for formatted_conflict in formatted_conflicts:
            log.info(f'RESULT {formatted_conflict}')

        find_conflict_result = {
            'conflicts': formatted_conflicts,
            'time': timer.elapsed(),
        }
        self.log_result(find_conflict_result=find_conflict_result)


        return conflicts

    def _find_conflics(self, cnf: CNF, assumptions: list[int]) -> CNF:
        if any(assumption < 0 for assumption in assumptions):
            raise ValueError('only positive assumptions are supported')

        solver = xor_cnf_as_cryptominisat_solver(cnf)
        conflicts = CNF(nvars=cnf.nvars)

        log.info(f'solving CNF with #Clauses: {len(self.cnf._clauses)}, #XORs: {len(self.cnf._xor_clauses)}, #Vars: {self.cnf.nvars}, #Assumptions: {len(assumptions)}')

        with Timer() as timer:
            is_sat, _ = solver.solve(assumptions)

        if is_sat:
            # no extra conflicts
            return conflicts

        conflict = Clause(solver.get_conflict())
        conflicts.add_clause(conflict)
        log.info(f'conflict: {self.format_clause(conflict)}')

        for conflicting_var in conflict:
            new_assumptions = [assumption for assumption in assumptions if assumption != -conflicting_var]
            assert new_assumptions != assumptions, f'conflicting variable ({conflicting_var}) not found in assumptions ({assumptions})'
            conflicts += self._find_conflics(cnf, new_assumptions)

        return conflicts



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


    def count_probability(self, epsilon: float, delta: float, fixed_key: bool=False, fixed_tweak: bool=False, verbosity: int=2) -> CountResult:
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

        seed = int.from_bytes(os.urandom(4), 'little')
        with Timer() as timer:
            num_solutions = count_solutions(cnf, epsilon, delta, verbosity=verbosity, sampling_set=sampling_set, seed=seed)

        prob = num_solutions / (1<<denominator_log2)

        log.info(f'RESULT {log_str}: {fmt_log2(prob)}')

        count_result = {
            'probability': prob,
            'epsilon': epsilon,
            'delta': delta,
            'key' : key_str,
            'tweak': tweak_str,
            'time': timer.elapsed(),
            'seed': seed,
        }
        self.log_result(count_result=count_result)

        return CountResult(prob, key_bits, tweak_bits)

    def get_tweak_or_key_variables(self, get_key: bool, get_tweak: bool) -> list[int]:
        variables = []
        if get_key:
            variables += self.key.flatten().tolist()
        if get_tweak:
            variables += self.tweak.flatten().tolist()
        return variables

    def count_tweakey_space(self, epsilon, delta, count_key: bool=True, count_tweak: bool=True, verbosity: int=2):
        """
        Use model counting to count the number of tweakeys for which the
        characteristic is not impossible.
        """
        sampling_set = self.get_tweak_or_key_variables(count_key, count_tweak)

        name = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]

        seed = int.from_bytes(os.urandom(4), 'little')
        with Timer() as timer:
            num_keys = count_solutions(self.cnf, epsilon, delta, verbosity=verbosity, sampling_set=sampling_set, seed=seed)

        log_num_keys = fmt_log2(num_keys)
        log.info(f'RESULT {name} space: {log_num_keys}, {epsilon=}, {delta=}')

        count_tweakey_result = {
            'num_keys': num_keys,
            'epsilon': epsilon,
            'delta': delta,
            'count_key': count_key,
            'count_tweak': count_tweak,
            'time': timer.elapsed(),
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
        determine the size of the affine hull of the tweakey space
        """
        sampling_set_list = np.array(self.get_tweak_or_key_variables(count_key, count_tweak))
        sampling_set = set(sampling_set_list)

        # one extra sample because the affine space has an offset as well
        count_initial_samples = len(sampling_set) + 1
        initial_samples: list[GF2] = []

        start_time = time.monotonic()
        name = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]
        with ThreadPoolExecutor(max_workers=_available_cpus()) as executor:
            def task(_index):
                raw_model = self._solve(log_result=False)
                return GF2(raw_model[sampling_set_list])

            samples = executor.map(task, range(count_initial_samples))
            for sample in tqdm(samples, total=count_initial_samples, desc=f'gathering valid {name}s'):
                initial_samples.append(sample)

        A, b = None, None
        while A is None or b is None:
            affine_space = affine_hull(initial_samples)
            log.info(f'gathered {name}s span affine space of dimension {affine_space.dimension()}')

            A, b = affine_space.as_equation_system()

            extra_vars = range(self.cnf.nvars + 1, self.cnf.nvars + len(A) + 1)
            extra_constraints = XorCNF()
            extra_constraints.nvars = self.cnf.nvars + len(A)

            for i, eq, rhs in zip(count(), A, b):
                variables, = np.nonzero(eq)
                var_idxes = sampling_set_list[variables]

                # create xors with extra variables that track whether the xor is satisfied
                extra_constraints += XorCNF.create_xor(*var_idxes[:, np.newaxis], [extra_vars[i]], rhs=[rhs])

            # at least one of the xors must be violated
            extra_constraints.add_clauses(list(extra_vars) + [0])
            try:
                log.info('solving for counterexample')
                raw_model = self._solve(self.cnf + extra_constraints, log_result=False)
                log.info('found counterexample -> trying again')
                sample = raw_model[sampling_set_list]
                initial_samples.append(GF2(sample))
            except UnsatException as e:
                log.info(f'RESULT no counterexample found -> conditions on {name} are necessary')
                break
        end_time = time.monotonic()

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
        sampling_set_list = self.get_tweak_or_key_variables(count_key, count_tweak)
        sampling_set = set(sampling_set_list)

        name = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]

        solver = xor_cnf_as_cryptominisat_solver(self.cnf)

        log.info(f'solving with CryptoMiniSat #Clauses: {len(self.cnf._clauses)}, #XORs: {len(self.cnf._xor_clauses)}, #Vars: {self.cnf.nvars}')

        count_sat = 0
        count_unsat = 0

        key_cnf = CNF([], nvars=self.cnf.nvars)

        prev_keys_update = 0.0
        with Timer() as timer:
            pbar = tqdm(range(trials))
            for i in pbar:
                sampling_new = [x * (-1)**random.randint(0, 1) for x in sampling_set_list]
                is_sat, _ = solver.solve(self.sbox_assumptions.ravel().tolist() + sampling_new)
                count_sat += is_sat
                count_unsat += not is_sat

                if not is_sat:
                    conflict = Clause(solver.get_conflict())
                    if conflict not in key_cnf:
                        key_cnf.add_clause(conflict)
                        tqdm.write(self.format_clause(conflict))

                if time.perf_counter() - prev_keys_update >= 0.1:
                    prev_keys_update = time.perf_counter()
                    pbar.set_description(f'valid {name}s: {count_sat}/{i + 1}')

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
            'time': timer.elapsed(),
        }

        self.log_result(count_tweakeys_sat_result=count_tweakeys_sat_result)

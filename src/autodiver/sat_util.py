from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from dataclasses import dataclass
import logging
import tempfile
import os
import subprocess as sp
import shutil

from sat_toolkit.formula import XorCNF, CNF, Truthtable
from tqdm import tqdm
import re
import pycryptosat
import numpy as np

from .util import fmt_log2
from .types import UnsatException


log = logging.getLogger(__name__)
_cnf_cache: dict[tuple[bytes, bytes], CNF] = {}


def truthtable_to_cnf(tt: Truthtable) -> CNF:
    cache_key = (tt.on_set.astype(np.int32).tobytes(), tt.dc_set.astype(np.int32).tobytes())
    try:
        # don't return the cached version, because it's mutable
        return CNF(_cnf_cache[cache_key])
    except KeyError:
        pass

    cnf = tt.to_cnf()
    _cnf_cache[cache_key] = cnf

    # don't return the cached version, because it's mutable
    return CNF(cnf)

def lut_to_cnf(lut: np.ndarray) -> CNF:
    truthtable = Truthtable.from_lut(lut.T.flatten())
    return truthtable_to_cnf(truthtable)

def xor_cnf_as_cryptominisat_solver(xor_cnf: XorCNF) -> pycryptosat.Solver:
    solver = pycryptosat.Solver()
    solver.add_clauses(xor_cnf._clauses)
    for xor_clause in xor_cnf._xor_clauses:
        rhs = int(np.prod(np.sign(xor_clause))) > 0
        pos_clause = np.abs(xor_clause).tolist()
        solver.add_xor_clause(pos_clause, rhs=rhs)
    return solver


class XorClause():
    def __init__(self, literals: Iterable[Integral]):
        self.literals: set[Integral] = set()
        for literal in literals:
            self.add(literal)

    def xor_constant(self, constant: bool|int):
        if constant not in (0, 1):
            raise ValueError("Constant must be 0/1 or False/True")

        if len(self.literals) == 0:
            raise ValueError("Cannot xor constant with empty clause")

        if constant:
            lit = self.literals.pop()
            self.literals.add(-lit)


    def add(self, literal: Integral):
        if literal in self.literals:
            self.literals.remove(literal)
        else:
            self.literals.add(literal)

    def to_cnf(self):
        rhs = 0
        args = []
        for literal in self.literals:
            rhs ^= literal < 0
            args.append([abs(literal)])

        return CNF.create_xor(*args, rhs=rhs)

    def __ixor__(self, other: Integral|XorClause):
        if isinstance(other, XorClause):
            for literal in other.literals:
                self ^= literal

        assert isinstance(other, Integral)

        if other in self.literals:
            self.literals.remove(other)
        else:
            self.literals.add(other)

        return self

    def __xor__(self, other: Integral|XorClause):
        result = XorClause(self.literals)
        result ^= other
        return result

    def __repr__(self):
        return f"XorClause({self.literals!r})"


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


_warning_emited = False
def count_solutions(cnf: XorCNF, epsilon: float, delta: float, verbosity: int=2, sampling_set: list[int] | None=None, seed: int|None=None) -> int:
    global _warning_emited

    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')

    sampling_set_log = f" over {len(sampling_set)} variables" if sampling_set is not None else ""
    log.info(f'counting solutions to cnf with {cnf.nvars} variables, {cnf.nclauses} clauses, and {cnf.nxor_clauses} xor clauses{sampling_set_log}, {epsilon=}, {delta=}')

    approxmc = shutil.which('approxmc')
    if approxmc is None:
        if not _warning_emited:
            log.warning('approxmc not found in $PATH, falling back to pyapproxmc. Cannot provide progress updates.')
            _warning_emited = True

        from pyapproxmc import Counter
        ctr = Counter(seed=seed, epsilon=epsilon, delta=delta)
        ctr.add_clauses(cnf.to_cnf())

        # add tautological clause to ensure the number of variables is correct
        ctr.add_clause([cnf.nvars, -cnf.nvars])

        if sampling_set is not None:
            solution_count, hash_count = ctr.count(sampling_set)
        else:
            solution_count, hash_count = ctr.count()

        return solution_count * 2**hash_count


    # create temporary file for cnf
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf') as f:
        if sampling_set is not None:
            sampling_set_str = ' '.join(str(x) for x in sampling_set) + ' 0'
            f.write(f'c ind {sampling_set_str}\n')

        # ApproxMC does not support XOR clauses, so we convert to a CNF
        f.write(cnf.to_cnf().to_dimacs())
        f.flush()

        # run approxmc
        args = [approxmc, f'--seed={seed}', f'--{epsilon=}', f'--{delta=}', '--sparse=1', f'--verb={verbosity}', f.name]

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




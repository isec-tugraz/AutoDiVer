from __future__ import annotations
from typing import Any
from functools import reduce, lru_cache
from itertools import product
import numpy as np
from os import path
from .constants import *
from .read_sol import *
from ..sat_util import XorClause
class LfsrState:
    def __init__(self, name: str, connection_poly: list[int], vars:np.ndarray[Any, np.dtype[np.int32]]):
        """
        connection_poly: list of exponents of the connection polynomial
        """
        assert connection_poly[0] == connection_poly[-1] == 1
        self.name = name
        self.vars = vars
        self.degree = len(connection_poly) - 1
        self.poly = np.array(connection_poly)
        self.constraints_generated = False
    def __repr__(self):
        return f'LfsrState({self.name!r}, {self.poly!r})'
    def get_constraints(self, ref_indices: range|None = None):
        if ref_indices is None:
            start = (len(self.vars) - self.degree) // 2
            ref_indices = range(start, start + self.degree)
        start = ref_indices.start
        for i in ref_indices:
            self.get_bit(i)
        constraints = []
        self.constraints_generated = True
        for i, _ in enumerate(self.vars):
            if i in ref_indices:
                continue
            offset = i - start
            mask = self.get_bit_mask(offset)
            variables = [self.vars[start + i] for i, e in enumerate(mask) if e]
            clause = XorClause([self.vars[i]])
            for var in variables:
                clause ^= var
            constraints.append(clause)
        return constraints
    def get_bit(self, index: int) -> np.int32:
        assert index in range(len(self.vars)), f'index {index} out of range(0, {len(self.vars)})'
        return self.vars[index]
    def get_bit_range(self, start_idx, numbits=8) -> np.ndarray[Any, np.dtype[np.int32]]:
        assert start_idx in range(len(self.vars))
        assert start_idx + numbits - 1 in range(len(self.vars))
        return self.vars[start_idx:start_idx + numbits]
    def __getitem__(self, index):
        return self.vars[index]
    @lru_cache(maxsize=None)
    def get_bit_mask(self, index: int) -> np.ndarray:
        """
        returns a mask over the indices [0, ..., len(self.poly) - 1] that can
        be used to calculate the bit at index `index`.
        """
        result = np.zeros(self.degree, dtype=int)
        if index in range(self.degree):
            result[index] = 1
            return result
        assert self.poly[0] == self.poly[self.degree] == 1
        if index >= self.degree:
            for i in range(self.degree):
                if self.poly[i] == 1:
                    offset = self.degree - i
                    result ^= self.get_bit_mask(index - offset)
            return result
        if index < 0:
            for i in range(1, self.degree + 1):
                if self.poly[i] == 1:
                    result ^= self.get_bit_mask(index + i)
            return result
        raise AssertionError('unreachable')
def get_solution_set(sbox: np.ndarray, in_delta, out_delta):
    in_vals = np.arange(len(sbox), dtype=np.uint8)
    in_vals = in_vals[sbox[in_vals] ^ sbox[in_vals ^ in_delta] == out_delta]
    return in_vals
def precisedelta(time_range: float):
    res = ''
    days = int(time_range / 86_400)
    hours = int(time_range / 3_600) % 24
    minutes = int(time_range / 60) % 60
    seconds = int(time_range) % 60
    millis = int(time_range * 1_000) % 1000
    # micros = int(time_range * 1_000_000) % 1000
    # nanos = int(time_range * 1_000_000_000) % 1000
    if days > 0:
        res += f'{days} days '
    if hours > 0:
        res += f'{hours} h '
    if minutes > 0:
        res += f'{minutes} m '
    if seconds > 0:
        res += f'{seconds} s '
    if millis > 0:
        res += f'{millis} ms '
    # if micros > 0:
    #      res += f'{micros} us '
    # if nanos > 0:
    #      res += f'{nanos} ns '
    res = res.strip()
    if res == '':
        res = '0 s'
    return res
#!/usr/bin/env python3
# from typing import *
from ast import literal_eval
from collections import defaultdict
from dataclasses import dataclass
import io
import re
import numpy as np
def transform_vars(var: dict) -> np.array:
    numrounds = max(round_nr for _, _, round_nr in var.keys()) + 1
    result = np.zeros(shape=(numrounds, 4, 4), dtype=int)
    result[:] = -1
    for (row, col, round_nr), value in var.items():
        result[round_nr, row, col] = value
    assert not np.any(result == -1)
    return result
def _read_milp_sol(f: io.TextIOWrapper):
    variables = defaultdict(lambda : dict())
    for line in f:
        if line.startswith('#'):
            continue
        varname, val = line.split()
        val = literal_eval(val)
        varname = varname.replace('_', ' ')
        try:
            idx = varname.index('(')
        except ValueError:
            idx = varname.index('[')
        varname, idx = varname[:idx], varname[idx:]
        # idx == (row, col, round_nr)
        idx = tuple(literal_eval(idx))
        variables[varname][idx] = val
    return dict(variables)
def read_milp_sol(f: str|io.TextIOWrapper):
    if isinstance(f, io.TextIOWrapper):
        return _read_milp_sol(f)
    with open(f, 'r') as f:
        return _read_milp_sol(f)
def _parse_nldtool_round(rnd: int, lines: list[str]):
    res = defaultdict(lambda: [None] * len(lines))
    for line in lines:
        parts = re.split(r' (?=[A-Za-z0-9]+[0-9]:)', line)
        parts = [part.split(': ') for part in parts]
        parts = {k: v for k, v in parts}
        for k, v in parts.items():
            k, idx = k[:-1], int(k[-1:])
            assert res[k][idx] is None
            res[k][idx] = v
    return dict(res)
def _read_nldtool_sol(f: io.TextIOWrapper) -> list[dict[str, list[str]]]:
    result = []
    lines_for_round = []
    prev_round = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        rnd, line = line.split(maxsplit=1)
        rnd = int(rnd)
        if rnd != prev_round:
            if prev_round is not None:
                if prev_round != len(result):
                    raise ValueError(f"expected differences for round {len(result)} got {prev_round}")
                result.append(_parse_nldtool_round(prev_round, lines_for_round))
            prev_round = rnd
            lines_for_round = []
        lines_for_round.append(line.strip())
    if prev_round is not None:
        if prev_round != len(result):
            raise ValueError(f"expected differences for round {len(result)} got {prev_round}")
        result.append(_parse_nldtool_round(prev_round, lines_for_round))
    return result
def read_nldtool_sol(f: str|io.TextIOWrapper) -> list[dict[str, list[str]]]:
    if isinstance(f, io.TextIOWrapper):
        return _read_nldtool_sol(f)
    with open(f, 'r') as f:
        return _read_nldtool_sol(f)
def nldtool_state_to_byte_differences(state: list[str]) -> np.ndarray:
    assert len(state[0]) % 8 == 0
    rows, cols = len(state), len(state[0]) // 8
    res = np.zeros((rows, cols), dtype=np.uint8)
    for row_idx, row in enumerate(state):
        for bit_idx, bit_cond in enumerate(row):
            if bit_cond == '-'or bit_cond == '0' or bit_cond == '1':
                pass
            elif bit_cond == 'x' or bit_cond == 'n' or bit_cond == 'u':
                res[row_idx, bit_idx // 8] |= 1 << (7 - (bit_idx % 8))
            else:
                raise ValueError(f"unknown bit condition: '{bit_cond}'")
    return res
def convert_nldtool_characteristic_to_bitwise(nldtool_characteristic: list[dict[str, list[str]]]):
    result: list[dict[str, list[list[int]]]] = []
    for d in nldtool_characteristic:
        result.append({ k: nldtool_state_to_byte_differences(v) for k, v in d.items() })
    return result
@dataclass
class TruncatedCharacteristic:
    filename: str
    sbox: np.ndarray
    rkey: np.ndarray
    tbox: np.ndarray
    tkey: np.ndarray
    unknown: np.ndarray
    forget: np.ndarray
    numrounds: int
    shape: tuple[int, int, int]
    def __init__(self, byte_char_file: str):
        self.filename = byte_char_file
        milp_char = read_milp_sol(byte_char_file)
        self.sbox = transform_vars(milp_char['sbox'])[:-1]
        self.rkey = transform_vars(milp_char['rkey'])[:-1]
        self.tbox = transform_vars(milp_char['tbox'])[:-1]
        self.tkey = transform_vars(milp_char['tkey'])[:-1]
        self.unknown = transform_vars(milp_char['unkn'])[:-1]
        self.forget = transform_vars(milp_char['forg'])[:-1]
        if not (self.sbox.shape == self.tbox.shape == self.tkey.shape == self.unknown.shape == self.forget.shape):
            raise ValueError('shape mismatch')
        self.shape = self.sbox.shape
        self.numrounds = self.shape[0]
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    read_milp_sol(args.filename)
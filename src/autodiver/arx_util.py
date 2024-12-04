"""
utility functions to model ARX ciphers
"""
from __future__ import annotations

from typing import Any
from sat_toolkit.formula import CNF, Truthtable
import numpy as np

import sys

_full_adder_cache = dict()


def _model_full_adder(input_delta: tuple[int, int, int], carry_delta: int) -> CNF:
    """
    create a CNF model that asserts a differential transition for a full adder.
    input[0..2] correspond to variables 1, 2, and 3
    sum_out corresponds to variable 4
    carry_out corresponds to variable 5
    """
    assert all(x in (0, 1) for x in input_delta)
    assert carry_delta in (0, 1)

    on_indices = []
    dc_indices = []

    a_diff, b_diff, c_diff = input_delta

    for idx in range(32):
        a = (idx >> 0) & 1
        b = (idx >> 1) & 1
        c = (idx >> 2) & 1
        sum_out = (idx >> 3) & 1
        carry_out = (idx >> 4) & 1


        if (a ^ b ^ c ^ sum_out) != 0:
            continue

        carry_1 = (a + b + c) >> 1
        carry_2 = ((a ^ a_diff) + (b ^ b_diff) + (c ^ c_diff)) >> 1


        # is valid carry?
        if (carry_1 == carry_out) and (carry_1 ^ carry_2 == carry_delta):
            on_indices.append(idx)
            continue


    tt = Truthtable.from_indices(numbits=5, on_indices=np.array(on_indices, int))

    cnf = tt.to_cnf()
    return cnf

def model_full_adder(input_delta: tuple[int, int, int], carry_delta: int) -> CNF:
    """
    create a CNF model that asserts a differential transition for a full adder.
    input[0..2] correspond to variables 1, 2, and 3
    sum_out corresponds to variable 4
    carry_out corresponds to variable 5
    """

    key = (input_delta, carry_delta)
    if key in _full_adder_cache:
        return _full_adder_cache[key].copy()

    cnf = _model_full_adder(input_delta, carry_delta)
    _full_adder_cache[key] = cnf
    return cnf.copy()

def model_modular_addition(input_delta: tuple[int, int], output_delta: int, numbits: int, model_assumptions: bool=False) -> CNF:
    """
    create a CNF model that asserts a differential transition for a modular addition.
    input[0] correspond to variables 1 -- numbits
    input[1] correspond to variables numbits+1 -- 2*numbits
    output corresponds to variables 2*numbits+1 -- 3*numbits
    the temporary variables for the carrys correspond to variables 3*numbits+1 -- 4*numbits
    assumption (if model_assumptions) corresponds to variables 4*numbits+1 -- 5*numbits

    all variables are little-endian indexed, i.e. variable 1 corresponds to the least significant bit
    """

    if input_delta[0] >= 1 << numbits:
        raise ValueError("input_delta[0] is too large")
    if input_delta[1] >= 1 << numbits:
        raise ValueError("input_delta[0] is too large")
    if output_delta >= 1 << numbits:
        raise ValueError("output_delta is too large")

    carry_in_delta = (input_delta[0] ^ input_delta[1] ^ output_delta)

    # assert (carry_in_delta & 1) == 0
    if (carry_in_delta & 1) != 0:
        print(f'invalid transition: {input_delta[0]:04x}, {input_delta[1]:04x} -> {output_delta:04x}')
        raise ValueError("invalid transition: carry_in_delta[0] is not zero")


    carry_out_delta = carry_in_delta >> 1


    invars0 = np.arange(1, numbits+1)
    invars1 = np.arange(numbits+1, 2*numbits+1)
    outvars = np.arange(2*numbits+1, 3*numbits+1)
    carryvars = np.arange(3*numbits+1, 4*numbits+1)

    numvars = 5 * numbits if model_assumptions else 4 * numbits
    cnf = CNF(nvars=numvars)
    cnf.add_clause([-carryvars[0]])

    for i in range(numbits):
        local_cnf = None
        if i < numbits - 1:
            in_delta1 = (input_delta[0] >> i) & 1
            in_delta2 = (input_delta[1] >> i) & 1
            in_delta_3 = (carry_in_delta >> i) & 1
            local_carry_out_delta = (carry_out_delta >> i) & 1

            temp_cnf = model_full_adder((in_delta1, in_delta2, in_delta_3), local_carry_out_delta)
            new_vars = np.array([0, invars0[i], invars1[i], carryvars[i], outvars[i], carryvars[i+1]])
            local_cnf = temp_cnf.translate(new_vars)
        else:
            assert i == numbits - 1
            local_cnf = CNF.create_xor([invars0[i]], [invars1[i]], [carryvars[i]], [outvars[i]])

        if model_assumptions:
            local_cnf = local_cnf.implied_by(4 * numbits + 1 + i)

        cnf += local_cnf

    return cnf

def _eq(x, y, z):
    return (~x ^ y) & (~x ^ z)


if sys.version_info >= (3, 10):
    _hamming_weight = np.vectorize(lambda x: x.bit_count(), otypes=(int,))
else:
    _hamming_weight = np.vectorize(lambda x: bin(x).count('1'), otypes=(int,))


def modular_addition_probability(delta_in1: np.ndarray[Any, np.dtype[np.uint64]], delta_in2: np.ndarray[Any, np.dtype[np.uint64]], delta_out: np.ndarray[Any, np.dtype[np.uint64]], numbits: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    calculate the probability of a differential transition for a modular addition
    """
    mask = (1 << numbits) - 1
    assert np.all(delta_in1 & mask == delta_in1)
    assert np.all(delta_in2 & mask == delta_in2)
    assert np.all(delta_out & mask == delta_out)

    carry_in = delta_in1 ^ delta_in2 ^ delta_out

    is_valid = _eq(delta_in1 << 1, delta_in2 << 1, delta_out << 1) & (carry_in ^ delta_in2 << 1) & mask

    weight = _hamming_weight(~_eq(delta_in1, delta_in2, delta_out) & (mask >> 1))
    prob = 1.0 / (1 << weight)

    prob[is_valid & mask != 0] = 0

    return prob

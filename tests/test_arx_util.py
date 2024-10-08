from __future__ import annotations

import pytest

from autodiver.arx_util import model_full_adder, model_modular_addition


def test_simple():
    cnf = model_full_adder((0, 0, 0), 0)
    tt = cnf.to_truthtable()

    on_set = tt.on_set
    print(on_set)

    for idx in range(32):
        a = (idx >> 0) & 1
        b = (idx >> 1) & 1
        c = (idx >> 2) & 1
        sum_out = (idx >> 3) & 1
        carry_out = (idx >> 4) & 1

        is_valid = idx in on_set

        sum_valid = (a ^ b ^ c) == sum_out
        carry_valid = (a + b + c) >> 1 == carry_out

        assert is_valid == (sum_valid and carry_valid), f"idx={idx} a={a} b={b} c={c} sum_out={sum_out} carry_out={carry_out}"


@pytest.mark.parametrize("delta_str, carry_delta", [
    ("000", 0), ("100", 0), ("010", 0), ("001", 0),
    ("110", 0), ("101", 0), ("011", 0), ("111", 0),
    ("000", 1), ("100", 1), ("010", 1), ("001", 1),
    ("110", 1), ("101", 1), ("011", 1), ("111", 1),
])
def test_nonzero_diff(delta_str: str, carry_delta: int):
    input_delta = (int(delta_str[0]), int(delta_str[1]), int(delta_str[2]))

    cnf = model_full_adder(input_delta, carry_delta).minimize_espresso()
    tt = cnf.to_truthtable()
    print(cnf)

    on_set = tt.on_set

    for idx in range(32):
        a = (idx >> 0) & 1
        b = (idx >> 1) & 1
        c = (idx >> 2) & 1
        sum_out = (idx >> 3) & 1
        carry_out = (idx >> 4) & 1

        sum_ref = (a ^ b ^ c)
        carry_ref_1 = (a + b + c) >> 1
        carry_ref_2 = ((a ^ input_delta[0]) + (b ^ input_delta[1]) + (c ^ input_delta[2])) >> 1

        is_valid = idx in on_set

        sum_valid = sum_ref == sum_out
        carry_valid = carry_ref_1 == carry_out
        carry_delta_valid = carry_ref_1 ^ carry_ref_2 == carry_delta

        assert is_valid == (sum_valid and carry_valid and carry_delta_valid), f"idx={idx} a={a} b={b} c={c} sum_out={sum_out} carry_out={carry_out}"

@pytest.mark.parametrize("numbits, in1_delta, in2_delta, out_delta", [
    (32, 0, 0, 0),
    (32, 1, 1, 0),
    (32, 1, 0, 0xFFFFFFFF),
    (32, 0x100, 0, 0xFFFFFF00),
    (32, 0x100, 0x400, 0xFFFFFF00),
    (32, 0x100, 0xFFFFFE00, 0xFFFFFF00),
    (32, 1, 0x100, 0xFF),
])
def test_modular_add(numbits, in1_delta: int, in2_delta: int, out_delta: int):
    cnf = model_modular_addition((in1_delta, in2_delta), out_delta, numbits)

    seed = 0x18a37a02
    args = ['cryptominisat5', f'--random={seed}', '--polar=rnd']

    sat_result = cnf.solve_dimacs(args)
    assert sat_result[0] == True
    result = sat_result[1]

    inp1 = sum(int(result[1 + i]) << i for i in range(numbits))
    inp2 = sum(int(result[numbits + 1 + i]) << i for i in range(numbits))
    out = sum(int(result[2*numbits + 1 + i]) << i for i in range(numbits))

    print(f'  {inp1:8x} ^ {in1_delta:8x} = {inp1 ^ in1_delta:8x}')
    print(f'+ {inp2:8x} ^ {in2_delta:8x} = {inp2 ^ in2_delta:8x}')
    print(f'  {"-" * 8}-|-{"-" * 8}-|-{"-" * 8}')
    print(f'  {out:8x} ^ {out_delta:8x} = {out ^ out_delta:8x}')

    assert (inp1 + inp2) & ((1 << numbits) - 1) == out
    assert (inp1 ^ in1_delta) + (inp2 ^ in2_delta) & ((1 << numbits) - 1) == out ^ out_delta

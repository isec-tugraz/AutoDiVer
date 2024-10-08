from __future__ import annotations

from typing import Any

import numpy as np

ALPHA_MAP = {
    16: 7,
    24: 8,
    32: 8,
    48: 8,
    64: 8,
}

BETA_MAP = {
    16: 2,
    24: 3,
    32: 3,
    48: 3,
    64: 3,
}

def _rotr_value(value: int, shift: int, bitwidth: int) -> int:
    assert shift < bitwidth
    return (value >> shift) | (value << (bitwidth - shift)) & ((1 << bitwidth) - 1)

def _rotl_value(value: int, shift: int, bitwidth: int) -> int:
    return _rotr_value(value, bitwidth - shift, bitwidth)


def rotr_speck(value: int, bitwidth: int) -> int:
    amount = ALPHA_MAP[bitwidth]
    return _rotr_value(value, amount, bitwidth)

def rotl_speck(value: int, bitwidth: int) -> int:
    amount = BETA_MAP[bitwidth]
    return _rotl_value(value, amount, bitwidth)


def _rotr_np(value: np.ndarray[Any, np.dtype[np.uint64]], shift: int, bitwidth: int) -> np.ndarray[Any, np.dtype[np.uint64]]:
    assert shift < bitwidth

    result = (value >> shift) | (value << (bitwidth - shift)) & ((1 << bitwidth) - 1)
    return np.array(result, np.uint64)

def _rotl_np(value: np.ndarray[Any, np.dtype[np.uint64]], shift: int, bitwidth: int) -> np.ndarray[Any, np.dtype[np.uint64]]:
    return _rotr_np(value, bitwidth - shift, bitwidth)

def rotr_speck_np(value: np.ndarray[Any, np.dtype[np.uint64]], bitwidth: int) -> np.ndarray[Any, np.dtype[np.uint64]]:
    amount = ALPHA_MAP[bitwidth]
    return _rotr_np(value, amount, bitwidth)

def rotl_sepck_np(value: np.ndarray[Any, np.dtype[np.uint64]], bitwidth: int) -> np.ndarray[Any, np.dtype[np.uint64]]:
    amount = BETA_MAP[bitwidth]
    return _rotl_np(value, amount, bitwidth)


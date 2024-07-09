from __future__ import annotations
from math import log2
from typing import Any, Literal
import numpy as np


def fmt_log2(number: float, width: int=0) -> str:
    if number == 0:
        num_str = "0"
    else:
        num_str = f"2^{log2(number):.2f}"
    return num_str.rjust(width)


def get_ddt(sbox):
    ddt = np.zeros((len(sbox), len(sbox)), dtype=np.int16)
    for in_delta in range(len(sbox)):
        in_val = np.arange(len(sbox), dtype=sbox.dtype)
        out_delta = sbox[in_val] ^ sbox[in_val ^ in_delta]
        out_delta, counts = np.unique(out_delta, return_counts=True)
        ddt[in_delta, out_delta] = counts
    return ddt


class Model:
    def __init__(self, index_set: IndexSet, raw_model: np.ndarray[Any, np.dtype[np.uint8]], *, bitorder: Literal['big', 'little']='little'):
        self.raw_model = raw_model
        for fieldname in index_set._fieldnames:
            index_array = getattr(index_set, fieldname)
            if np.prod(index_array.shape) == 0:
                model = np.zeros_like(index_array, dtype=np.uint8)
            else:
                # make sure to handle negative indices correctly
                relevant_raw_model = raw_model[np.abs(index_array)] ^ (index_array < 0)
                packed = np.packbits(relevant_raw_model, axis=-1, bitorder=bitorder)
                if packed.shape[-1] in (1, 2, 4, 8):
                    model = packed.view(f'u{packed.shape[-1]}')
                    if model.shape[-1] == 1:
                        model = model[..., 0]
                else:
                    model = packed
                # model = packed[..., 0] if packed.shape[-1] == 1 else packed
            setattr(self, fieldname, model)


class IndexSet:
    numvars: int

    def __init__(self):
        self.numvars = 0
        self._fieldnames = set()

    def add_index_array(self, name: str, shape: tuple[int, ...]):
        length: int = np.prod(shape, dtype=np.int32) # type: ignore
        res = np.arange(self.numvars + 1, self.numvars + 1 + length, dtype=np.int32)
        res = res.reshape(shape)
        res.flags.writeable = False
        self.numvars += length
        self._fieldnames.add(name)
        setattr(self, name, res)

    def describe_idx_array(self, index_array: np.ndarray):
        """
        convenience function to return the underlying array name and unraveled
        index for each linear index in `index_array`.
        """
        variables = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        if np.any((index_array < 0) | (index_array >= self.numvars + 1)):
            raise IndexError("index out of bounds")
        res = [None] * np.prod(index_array.shape, dtype=int)
        for i, needle in enumerate(index_array.flatten()):
            if needle == self.numvars:
                res[i] = "1"
                continue
            for k, v in variables.items():
                if v.size == 0:
                    continue
                start, stop = v.flatten()[[0, -1]]
                rng = range(start, stop + 1)
                if needle in rng:
                    idx = np.unravel_index(rng.index(needle), v.shape)
                    res[i] = k + str(list(idx))
                    # res[i] = str(f'{idx[1]}{idx[2]}')
                    break
            else:
                assert False, f"index {needle} not found?"
        return np.array(res, dtype=object).reshape(index_array.shape)

    def format_clause(self, clause: np.ndarray[Any, np.dtype[np.int32]]) -> str:
        if len(clause) == 0:
            return "⊥"
        varnames = self.describe_idx_array(np.abs(clause))
        desc = [n if c > 0 else f"￢{n}"for n, c in zip(varnames, clause)]
        return " ⋁ ".join(desc)

    def format_cnf(self, cnf: np.ndarray[Any, np.dtype[np.int32]]) -> str:
        if len(cnf) == 0:
            return "⊤"
        return "\n ⋀ ".join(self.format_clause(clause) for clause in cnf)

    def get_model(self, raw_model: np.ndarray[Any, np.dtype[np.uint8]], *, bitorder: Literal['big', 'little']='little') -> Model:
        return Model(self, raw_model, bitorder=bitorder)

    def __repr__(self):
        res = f"{self.__class__.__name__}(\n"
        fieldnames_len = max(len(name) for name in self._fieldnames)
        for fieldname in self._fieldnames:
            field = getattr(self, fieldname)
            if field.shape == (0,) or field.shape == ():
                res += f"  {fieldname.ljust(fieldnames_len)} = {field!r},\n"
                continue
            min_val = field.ravel()[0]
            max_val = field.ravel()[-1]
            total_len = np.prod(field.shape)
            if max_val + 1 - min_val == total_len and np.all(field == np.arange(min_val, max_val + 1).reshape(field.shape)):
                res += f"  {fieldname.ljust(fieldnames_len)} = np.arange({min_val}, {max_val + 1}).reshape({field.shape!r}),\n"
                continue
            res += f"  {fieldname.ljust(fieldnames_len)} = ...,\n"
        res += ")"
        return res

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
class IndexSet:
    numvars: int
    def __init__(self):
        self.numvars = 0
        self._fieldnames = set()
    def add_index_array(self, name: str, shape: Tuple[int, ...]):
        length = np.prod(shape, dtype=np.int32)
        res = np.arange(self.numvars + 1, self.numvars + 1 + length, dtype=np.int32)
        res = res.reshape(shape)
        res.flags.writeable = 0
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
    def __repr__(self):
        res = f"{self.__class__.__name__}(\n"
        for fieldname in self._fieldnames:
            field = getattr(self, fieldname)
            min_val = field.ravel()[0]
            max_val = field.ravel()[-1]
            res += f"  {fieldname} = np.arange({min_val}, {max_val + 1}).reshape({field.shape!r})\n"
        res += ")"
        return res
from __future__ import annotations

TYPE_CHECKING=False
if TYPE_CHECKING:
    from typing import Any, Self

from pathlib import Path
import logging

import numpy as np
import numpy.typing as npt


log = logging.getLogger(__name__)


class DifferentialCharacteristic():
    num_rounds: int
    sbox_in: np.ndarray[Any, np.dtype[np.uint8]]
    sbox_out: np.ndarray[Any, np.dtype[np.uint8]]

    file_path: Path|None

    ddt: np.ndarray
    rounds_from_to: tuple[int, int]

    @classmethod
    def load(cls, characteristic_path: Path) -> Self:
        raise NotImplementedError("this must be implemented by subclasses")

    @classmethod
    def load_txt(cls, characteristic_path: Path) -> Self:
        trail_list = []
        with open(characteristic_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                assert len(line) == 16 or len(line) == 32
                line_deltas = [int(l, 16) for l in line[::-1]]
                trail_list.append(line_deltas)

        trail = np.array(trail_list)
        if len(trail) % 2 != 0:
            log.error(f'expected an even number of differences in {characteristic_path!r}')
            raise ValueError(f'expected an even number of differences in {characteristic_path!r}')

        sbox_in = trail[0::2]
        sbox_out = trail[1::2]

        return cls(sbox_in, sbox_out, file_path=characteristic_path)

    @classmethod
    def load_npz(cls, characteristic_path: Path) -> Self:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']
        return cls(sbox_in, sbox_out, file_path=characteristic_path)

    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike, file_path: Path|None=None):
        self.sbox_in = np.array(sbox_in, dtype=np.uint8)
        self.sbox_out = np.array(sbox_out, dtype=np.uint8)
        self.file_path = file_path
        if self.sbox_in.shape != self.sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')

        ddt_probs = self.ddt[self.sbox_in, self.sbox_out]
        if np.any(ddt_probs == 0):
            invalid_sboxes = np.array(np.where(ddt_probs == 0)).T
            raise ValueError(f'invalid s-boxes: {invalid_sboxes}')


        self.num_rounds = len(self.sbox_in)

    def log2_ddt_probability(self):
        ddt = self.ddt
        ddt_prob = np.log2(ddt[self.sbox_in, self.sbox_out] / len(ddt)).sum()
        return ddt_prob

    def tikzify(self):
        raise NotImplementedError("this should be implemented by subclasses")


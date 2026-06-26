import os.path
import sys

from ..cipher_model import DifferentialCharacteristic
from .util import DDT, PERM, perm_nibble, perm_nibble_16, perm_nibble_16_inv
from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Any
from io import StringIO


_PREAMBLE = r"""
\documentclass[varwidth=20cm]{standalone}

\usepackage{warp}
 % alternative scaling for different paper sizes: warpscalelncs / warpscaletosc / warpscaleeprint
\tikzset{warpscale/.style={warpscaleprint}}

\begin{document}

  \begin{tikzpicture}[warpfig]
    \foreach \z[evaluate=\z as \zf using int(4*\z)] in {0,...,31} {
      \draw[gray] (\z,0) node[above] {\tiny\zf};
      \foreach \zb in {0,...,3} { \draw[gray] (\z+.25*\zb,0) -- +(0,-3pt); }
    }
  \end{tikzpicture}
    """.strip("\n")

_DOCUMENT_END = r"""
\end{document}
""".strip("\n")


class WarpCharacteristic(DifferentialCharacteristic):
    ddt: np.ndarray | None = DDT
    sbox_count = 16
    rounds_in: np.ndarray[Any, np.dtype[np.int32]] | None
    rounds_out: np.ndarray[Any, np.dtype[np.int32]] | None

    @classmethod
    def load(cls, characteristic_path: Path) -> "WarpCharacteristic":
        if os.path.splitext(characteristic_path)[1] == ".npz":
            result = cls.load_npz(characteristic_path)
        else:
            result = cls.load_txt(characteristic_path)
        result.verify_linear_layer()
        return result

    @classmethod
    def load_npz(cls, characteristic_path: Path) -> "WarpCharacteristic":
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']
            # rounds_in = f['rounds_in']
            # rounds_out = f['rounds_out']
        result = cls(sbox_in, sbox_out, None, None, file_path=characteristic_path)
        return result

    def save_npz(self, path: Path, cipher_name: str, stat_sat_search: tuple[float, int, int]|None, modeled_log_prob: int|None, rounding_mode: str):
        kwargs = {}
        if stat_sat_search:
            kwargs["stat_sat_search"] = stat_sat_search
        if modeled_log_prob:
            kwargs["modeled_log_prob"] = modeled_log_prob
        if self.rounds_in is not None:
            kwargs["rounds_in"] = self.rounds_in
        if self.rounds_out is not None:
            kwargs["rounds_out"] = self.rounds_out
        np.savez(path, sbox_in=self.sbox_in, sbox_out=self.sbox_out, cipher_name=cipher_name, num_rounds=self.num_rounds, log_probability=self.log2_ddt_probability(), rounding_mode=rounding_mode, **kwargs)

    @classmethod
    def load_from_model(cls, model):
        sbox_in = model.sbox_in
        sbox_out = model.sbox_out
        rounds_in = model.rounds_in
        rounds_out = model.rounds_out
        return cls(sbox_in, sbox_out, rounds_in, rounds_out, file_path=None)

    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike, rounds_in: np.ndarray[Any, np.dtype[np.int32]]|None=None, rounds_out: np.ndarray[Any, np.dtype[np.int32]]|None=None, file_path: Path|None=None):
        super().__init__(sbox_in, sbox_out, file_path)
        self.rounds_in = rounds_in
        self.rounds_out = rounds_out

    @staticmethod
    def _interleave(even: np.ndarray, odd: np.ndarray) -> np.ndarray:
        """Combine the 16 even and 16 odd nibbles into the 32-nibble state(s)."""
        state = np.empty((even.shape[0], 2 * even.shape[1]), dtype=np.uint8)
        state[:, 0::2] = even
        state[:, 1::2] = odd
        return state

    @property
    def round_in(self) -> np.ndarray:
        def assign_ok(lhs, rhs):
            return np.all((lhs == 0xFF) | (lhs == rhs))

        rnds = self.num_rounds
        round_in = np.full((rnds + 1, 32), 0xFF, np.uint8)
        before_perm = np.full((rnds, 32), 0xFF, np.uint8)

        # even inputs are given by s-box inputs
        assert np.all(round_in[:-1, 0::2] == 0xFF)
        round_in[:-1, 0::2] = self.sbox_in

        # propagate info
        assert assign_ok(before_perm[:-1, :], round_in[1:-1, PERM])
        before_perm[:-1, :] = round_in[1:-1, PERM]  # inverse perm

        assert assign_ok(before_perm[:, 0::2], self.sbox_in)
        before_perm[:, 0::2] = self.sbox_in       # even nibbles just propagate

        assert assign_ok(round_in[1:-1, PERM], before_perm[:-1, :])
        round_in[1:-1, PERM] = before_perm[:-1, :]  # propagate to following round

        # calculate odd nibbles of pt difference
        assert assign_ok(round_in[0, 1::2], self.sbox_out[0] ^ before_perm[0, 1::2])
        round_in[0, 1::2] = self.sbox_out[0] ^ before_perm[0, 1::2]

        # last round difference, after s-box xor (odd nibbles)
        assert assign_ok(before_perm[rnds - 1, 1::2], round_in[rnds - 1, 1::2] ^ self.sbox_out[rnds - 1])
        before_perm[rnds - 1, 1::2] = round_in[rnds - 1, 1::2] ^ self.sbox_out[rnds - 1]

        assert assign_ok(round_in[-1, PERM], before_perm[-1, :])
        round_in[-1, PERM] = before_perm[-1, :]

        # from IPython import embed; embed()
        return round_in

    @property
    def before_perm(self) -> np.ndarray:
        round_in = self.round_in
        before_perm = np.full((self.num_rounds, 32), 0xFF)
        before_perm[:] = round_in[1:, PERM]
        return before_perm

    def verify_linear_layer(self) -> None:
        """Verify sbox_in/sbox_out are consistent through WARP's linear layer.

        WARP's linear layer is the nibble permutation that maps each round's
        output state onto the next round's input state. This checks that
        ``perm_nibble(round_out[r]) == round_in[r + 1]`` for every round and
        raises :class:`ValueError` on a mismatch.
        """
        # return
        round_in = self.round_in
        for r in range(self.num_rounds - 1):
            expected_out = round_in[r + 1]
            assert np.all(round_in[r, ::2] == self.sbox_in[r])
            before_perm = round_in[r].copy()
            before_perm[1::2] ^= self.sbox_out[r]

            # print(f" round {r} ".center(80, '-'))
            # print("before perm       ", " ".join(f"{x:x}" for x in before_perm).replace("0", "."))
            # print("expected out[PERM]", " ".join(f"{x:x}" for x in expected_out[PERM]).replace("0", "."))
            assert np.all(before_perm == expected_out[PERM])

    def tikzify(self) -> str:
        round_in = self.round_in
        round_out = self.before_perm
        result = StringIO()
        print(_PREAMBLE, file=result)

        for round in range(self.num_rounds):
            print(f"  \\warpround{{{round + 1}}}{{", file=result)

            for idx in range(self.sbox_count):
                sboxinput = round_in[round][2 * idx]
                xorinput = round_in[round][2 * idx + 1]
                xoroutput = round_out[round][2 * idx + 1]

                if sboxinput != 0:
                    print(f"  \\marksboxes{{{idx}/{2*idx}/{PERM[2*idx]}}}", file=result)

                if xorinput != 0 and xoroutput != 0:
                    print(f"  \\markbranches{{{2*idx+1}/{PERM[2*idx + 1]}}}", file=result)
                else:
                    if xorinput != 0:
                        print(f"  \\markbranchinput{{{2*idx+1}/{idx}}}", file=result)

                    if xoroutput != 0:
                        print(f"  \\markbranchoutput{{{idx}/{2 * idx + 1}/{PERM[2*idx + 1]}}}", file=result)

            print(f"  }}", file=result)


        print(_DOCUMENT_END, file=result)
        return result.getvalue()

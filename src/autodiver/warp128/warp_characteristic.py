import os.path

from ..cipher_model import DifferentialCharacteristic
from .util import DDT, perm
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

    @classmethod
    def load(cls, characteristic_path: Path):
        if os.path.splitext(characteristic_path)[1] == ".npz":
            return cls.load_npz(characteristic_path)
        else:
            return cls.load_txt(characteristic_path)

    @classmethod
    def load_npz(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']
            rounds_in = f['rounds_in']
            rounds_out = f['rounds_out']
        return cls(sbox_in, sbox_out, rounds_in, rounds_out, file_path=characteristic_path)

    def save_npz(self, path: Path, cipher_name: str, stat_sat_search: tuple[float, int, int]|None, modeled_log_prob: int|None, rounding_mode: str):
        kwargs = {}
        if stat_sat_search:
            kwargs["stat_sat_search"] = stat_sat_search
        if modeled_log_prob:
            kwargs["modeled_log_prob"] = modeled_log_prob
        np.savez(path, sbox_in=self.sbox_in, sbox_out=self.sbox_out, rounds_in=self.rounds_in, rounds_out=self.rounds_out, cipher_name=cipher_name, num_rounds=self.num_rounds, log_probability=self.log2_ddt_probability(), rounding_mode=rounding_mode, **kwargs)

    @classmethod
    def load_from_model(cls, model):
        sbox_in = model.sbox_in
        sbox_out = model.sbox_out
        rounds_in = model.rounds_in
        rounds_out = model.rounds_out
        return cls(sbox_in, sbox_out, rounds_in, rounds_out, file_path=None)

    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike, rounds_in: np.ndarray[Any, np.dtype[np.int32]], rounds_out: np.ndarray[Any, np.dtype[np.int32]], file_path: Path|None=None):
        super().__init__(sbox_in, sbox_out, file_path)
        self.rounds_in = rounds_in
        self.rounds_out = rounds_out

    def tikzify(self) -> str:
        result = StringIO()
        print(_PREAMBLE, file=result)

        for round in range(self.num_rounds):
            print(f"  \\warpround{{{round + 1}}}{{", file=result)

            for idx in range(self.sbox_count):
                sboxinput = self.rounds_in[round][2*idx]
                xorinput = self.rounds_in[round][2*idx + 1]
                xoroutput = self.rounds_out[round][2*idx + 1]

                if sboxinput != 0:
                    print(f"  \\marksboxes{{{idx}/{2*idx}/{perm[2*idx]}}}", file=result)

                if xorinput != 0 and xoroutput != 0:
                    print(f"  \\markbranches{{{2*idx+1}/{perm[2*idx + 1]}}}", file=result)
                else:
                    if xorinput != 0:
                        print(f"  \\markbranchinput{{{2*idx+1}/{idx}}}", file=result)

                    if xoroutput != 0:
                        print(f"  \\markbranchoutput{{{idx}/{2 * idx + 1}/{perm[2*idx + 1]}}}", file=result)

            print(f"  }}", file=result)


        print(_DOCUMENT_END, file=result)
        return result.getvalue()



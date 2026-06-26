import numpy as np
from ..cipher_model import DifferentialCharacteristic
from pathlib import Path
from .constants import do_mix_cols, do_shift_rows,tweakey_mask, update_tweakey, DDT4, DDT8
import numpy.typing as npt
from typing import Any, Self
from io import StringIO

_PREAMBLE = r"""
\documentclass[a4paper,11pt]{article}

\usepackage{geometry}
\usepackage{parskip}
\usepackage[hidelinks]{hyperref}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usepackage{skinny}

\usetikzlibrary{calc,positioning,cipher}

\colorlet{diffcolor}{red}

\begin{document}

\pagestyle{empty}
\clearpage

\begin{figure}[p!]
  \newcommand{\hex}[1]{\texttt{#1}}
  \centering
  %\tikzsetnextfilename{skinny-char}
  \begin{tikzpicture}[cellopts/.append style={font=\scriptsize\ttfamily}, raster/.style={gray}, yscale=.8]

    \SkinnyInit{}{}{}{}""".strip("\n")

_DOCUMENT_END = r"""
  \end{tikzpicture}
\end{figure}
\end{document}
""".strip("\n")

class _SkinnyBaseCharacteristic(DifferentialCharacteristic):
    block_size = 0
    tweakeys: np.ndarray
    sbox_count = 16

    @classmethod
    def load(cls, characteristic_path: Path) -> Self:
        rounds = 100

        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']
            tweakeys = f['tweakeys']
        sbox_in = np.array(sbox_in, dtype=np.int8)[:rounds]
        sbox_out = np.array(sbox_out, dtype=np.int8)[:rounds]
        tweakeys = np.array(tweakeys, dtype=np.int8)[:rounds]

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')

        return cls(sbox_in, sbox_out, tweakeys, file_path=characteristic_path)


    @classmethod
    def load_from_model(cls, model):
        sbox_in = model.sbox_in[:-1]
        sbox_out = model.sbox_out
        if model.tweak.shape == (0,):
            tweakeys = np.zeros((sbox_out.shape[0], 3, 4, 4))
        else:
            tweakeys = model._round_tweakeys

        return cls(sbox_in, sbox_out, tweakeys, file_path=None)

    def save_npz(self, path: Path, cipher_name: str, stat_sat_search: tuple[float, int, int]|None, modeled_log_prob: int|None, rounding_mode: str):
        kwargs = {}
        if stat_sat_search:
            kwargs["stat_sat_search"] = stat_sat_search
        if modeled_log_prob:
            kwargs["modeled_log_prob"] = modeled_log_prob
        np.savez(path, sbox_in=self.sbox_in, sbox_out=self.sbox_out, tweakeys=self.tweakeys, cipher_name=cipher_name, num_rounds=self.num_rounds, log_probability=self.log2_ddt_probability(), rounding_mode=rounding_mode, **kwargs)


    @classmethod
    def load_empty_characteristic(cls, num_rounds) -> DifferentialCharacteristic:
        sbox_in = sbox_out = np.zeros((num_rounds, 4, 4), dtype=np.uint8)
        tweakeys = np.zeros((num_rounds, 3, 4, 4), dtype=np.uint8)
        return cls(sbox_in, sbox_out, tweakeys)


    def __init__(self, sbox_in: npt.ArrayLike, sbox_out: npt.ArrayLike, tweakeys: npt.ArrayLike, **kwargs):
        super().__init__(sbox_in, sbox_out, **kwargs)
        self.tweakeys = np.array(tweakeys, dtype=np.uint8)
        if self.tweakeys.shape != (self.num_rounds, 3, 4, 4):
            raise ValueError('tweakeys must have shape (num_rounds, 3, 4, 4)')


        self.num_rounds = len(self.sbox_in)

    def verify_linear_layer(self):
        # sanity check characteristic
        for i in range(len(self.tweakeys) - 1):
            if not np.all(self.tweakeys[i + 1] == update_tweakey(self.tweakeys[i], self.block_size)):
                raise ValueError(f'tweakey update check failed at round {i}')
            rtk = np.bitwise_xor.reduce(self.tweakeys[i], axis=0) & tweakey_mask
            if not np.all(self.sbox_in[i + 1] == do_mix_cols(do_shift_rows(self.sbox_out[i] ^ rtk))):
               raise ValueError(f'round update check failed at round {i}')

    def truncate_rounds(self, rounds_from_to: tuple[int, int]):
        super().truncate_rounds(rounds_from_to)
        self.tweakeys = self.tweakeys[rounds_from_to[0]:rounds_from_to[1] + 1]

    def tikzify(self) -> str:
        #print(self.tweakeys)
        result = StringIO()
        print(_PREAMBLE, file=result)

        for rnd in range(self.num_rounds):
            #print(self.sbox_in[rnd])
            #print(self.sbox_out[rnd])
            sbox_in = self.sbox_in[rnd]
            sbox_out = self.sbox_out[rnd]
            print(f"    \\SkinnyRoundTK{{", file=result, end="")

            # before sbox
            for i in range(4):
                for j in range(4):
                    if sbox_in[i][j] != 0:
                        print(f"\\FillCell[diffcolor!25]{{ss{str(i) + str(j)}}}\\Cell{{ss{str(i) + str(j)}}}{{{sbox_in[i][j]:02x}}}", file=result, end="")
            print("}", file=result)

            # tweak
            print("                  {", file=result, end="")
            rtk = np.bitwise_xor.reduce(self.tweakeys[rnd], axis=0) & tweakey_mask
            print(rtk)
            for i in range(2):
                for j in range(4):
                    if rtk[i][j] != 0:
                        print(f"\\FillCell[diffcolor!25]{{ss{str(i) + str(j)}}}\\Cell{{ss{str(i) + str(j)}}}{{{rtk[i][j]:02x}}}", file=result, end="")
            print(f"}}{{}}{{}}", file=result) # tweak - currently empty


            #print(f"}}\n    {{}}{{}}{{}}", file=result) # tweak - currently empty

            # after sbox
            print("                  {", file=result, end="")
            for i in range(4):
                for j in range(4):
                    if sbox_out[i][j] != 0:
                        print(f"\\FillCell[diffcolor!25]{{ss{str(i) + str(j)}}}\\Cell{{ss{str(i) + str(j)}}}{{{sbox_out[i][j]:02x}}}", file=result, end="")
            print("}", file=result)

            # after sbox and tweak
            print("                  {", file=result, end="")
            for i in range(4):
                for j in range(4):
                    cell = sbox_out[i][j] ^ rtk[i][j]
                    if cell != 0:
                        print(
                            f"\\FillCell[diffcolor!25]{{ss{str(i) + str(j)}}}\\Cell{{ss{str(i) + str(j)}}}{{{cell:02x}}}",
                            file=result, end="")
            print("}", file=result)

            state_shifted = np.zeros_like(sbox_out)
            # after shiftrows
            print("                  {", file=result, end="")
            for i in range(4):
                for j in range(4):
                    cell = sbox_out[i][j] ^ rtk[i][j]
                    if cell != 0:
                        state_shifted[i][(i + j) % 4] = cell
                        print(
                            f"\\FillCell[diffcolor!25]{{ss{str(i) + str((j + i) % 4)}}}\\Cell{{ss{str(i) + str((j + i) % 4)}}}{{{cell:02x}}}",
                            file=result, end="")

            print("}", file=result)


            state_mixed = np.zeros_like(sbox_out)
            state_shifted [1,:] ^= state_shifted[2,:]
            state_shifted[2,:] ^= state_shifted[0,:]

            state_mixed[0,:] = state_shifted[3,:] ^ state_shifted[2,:]
            state_mixed[3,:] = state_shifted[2,:]
            state_mixed[2,:] = state_shifted[1,:]
            state_mixed[1,:] = state_shifted[0,:]

            if rnd == self.num_rounds - 1:
                print(f"    \\SkinnyFin{{", file=result, end="")
            else:
                print(f"    \\SkinnyNewLine{{", file=result, end="")


            # after mixcols
            print("{", file=result, end="")
            for i in range(4):
                for j in range(4):
                    if state_mixed[i][j] != 0:
                        print(
                            f"\\FillCell[diffcolor!25]{{ss{str(i) + str(j)}}}\\Cell{{ss{str(i) + str(j)}}}{{{state_mixed[i][j]:02x}}}",
                            file=result, end="")

            print("}}", file=result)

        print(_DOCUMENT_END, file=result)
        return result.getvalue()

class Skinny128Characteristic(_SkinnyBaseCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint16]] = DDT8
    block_size = 128

class Skinny64Characteristic(_SkinnyBaseCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint16]] = DDT4
    block_size = 64

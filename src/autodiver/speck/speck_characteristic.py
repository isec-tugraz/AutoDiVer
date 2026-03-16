from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autodiver.cipher_model import DifferentialCharacteristic
from autodiver.arx_util import modular_addition_probability
from .speck_util import rotr_speck_np
from io import StringIO

_PREAMBLE = r"""
\documentclass[varwidth=20cm]{standalone}

\usepackage{speck, tugcolors}

\begin{document}

\renewcommand{\SpeckLeft}[1]{$X^0_{#1}$}
\renewcommand{\SpeckRight}[1]{$X^1_{#1}$}
\renewcommand{\SpeckKey}[1]{$\text{RK}^{#1}$}

\begin{tikzpicture}[>=latex,fillopts/.style={tugred},raster/.style={gray!50},rot/.append style={specklabelstyle},yscale=.95]
""".strip("\n")

_DOCUMENT_END = r"""\end{tikzpicture}
\end{document}""".strip("\n")

class SpeckCharacteristic(DifferentialCharacteristic):
    round_in: np.ndarray[Any, np.dtype[np.uint64]]
    add_in1: np.ndarray[Any, np.dtype[np.uint64]]
    add_in2: np.ndarray[Any, np.dtype[np.uint64]]
    add_out: np.ndarray[Any, np.dtype[np.uint64]]
    wordsize: int

    def __init__(self, round_in: np.ndarray, wordsize: int | None=None, file_path: Path|None=None):
        self.rounds_from_to = None
        self.file_path = file_path
        self.round_in = round_in
        self.num_rounds = len(round_in) - 1

        assert self.round_in.shape == (self.num_rounds + 1, 2)

        if wordsize and wordsize != self.wordsize:
            raise ValueError(f"expected wordsize {self.wordsize}, but got wordsize {wordsize}")

        self.add_in1 = rotr_speck_np(round_in[:-1, 0], self.wordsize)
        self.add_in2 = round_in[:-1, 1]
        self.add_out = round_in[1:, 0]


    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            wordsize = int(f['wordsize'])
            round_in = np.array(f['round_in'], dtype=np.uint64)

        return cls(round_in=round_in, wordsize=wordsize, file_path=characteristic_path)

    @classmethod
    def load_from_model(cls, model) -> DifferentialCharacteristic:
        round_in = model.round_in
        return cls(round_in=round_in, wordsize=None, file_path=None)

    @classmethod
    def load_empty_characteristic(cls, num_rounds) -> DifferentialCharacteristic:
        input_diffs = np.zeros((num_rounds + 1, 2), np.uint64)

        return cls(input_diffs)


    def truncate_rounds(self, rounds_from_to: tuple[int, int]):
        current_rounds = range(self.num_rounds)
        start, end = rounds_from_to

        assert start in current_rounds
        assert end in current_rounds

        self.round_in = self.round_in[start:end + 2]
        self.add_in1 = self.add_in1[start:end + 1]
        self.add_in2 = self.add_in2[start:end + 1]
        self.add_out = self.add_out[start:end + 1]
        self.rounds_from_to = rounds_from_to

        self.num_rounds = end + 1 - start
        assert self.num_rounds == len(self.add_in1) == len(self.add_in2) == len(self.add_out) == len(self.round_in) - 1

    def log2_ddt_probability(self):
        round_probs = modular_addition_probability(self.add_in1, self.add_in2, self.add_out, self.wordsize)
        return np.log2(round_probs).sum()

    def tikzify(self) -> str:
        result = StringIO()
        print(_PREAMBLE, file=result)
        print(f"\\SpeckInit[{self.wordsize*2}]", file=result)

        for rnd in range(self.num_rounds):
            print(f"\\SpeckRound{{{rnd + 1}}}", file=result)
            state = self.round_in[rnd]
            print(state)

            for i in range(2):
                print("{", file=result, end="")
                for bitidx in range(self.wordsize):
                    if state[i] & (1 << bitidx):
                        print(f"\\Fill{{s{bitidx}}}", file=result, end="")
                print("}", file=result)

            print("{", file=result, end="")
            for bitidx in range(self.wordsize):
                if state[0] & (1 << ((bitidx + 7) % self.wordsize)):
                    print(f"\\Fill{{s{bitidx}}}", file=result, end="")
            print("}", file=result)

            print("{", file=result, end="")
            for bitidx in range(self.wordsize):
                if state[1] & (1 << ((bitidx - 2) % self.wordsize)):
                    print(f"\\Fill{{s{bitidx}}}", file=result, end="")
            print("}", file=result)

        print(f"\\SpeckFinal{{{self.num_rounds + 1}}}", file=result)
        state = self.round_in[self.num_rounds]
        print(state)

        for i in range(2):
            print("{", file=result, end="")
            for bitidx in range(self.wordsize):
                if state[i] & (1 << bitidx):
                    print(f"\\Fill{{s{bitidx}}}", file=result, end="")
            print("}", file=result)



        print(_DOCUMENT_END, file=result)
        return result.getvalue()

class Speck32Characteristic(SpeckCharacteristic):
    wordsize = 16

class Speck48Characteristic(SpeckCharacteristic):
    wordsize = 24

class Speck64Characteristic(SpeckCharacteristic):
    wordsize = 32

class Speck96Characteristic(SpeckCharacteristic):
    wordsize = 48

class Speck128Characteristic(SpeckCharacteristic):
    wordsize = 64

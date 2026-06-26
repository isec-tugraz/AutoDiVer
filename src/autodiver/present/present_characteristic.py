from typing import Any
from io import StringIO

from .present_util import bit_perm, PRESENT_DDT
from ..cipher_model import DifferentialCharacteristic
from pathlib import Path

_PREAMBLE = r"""
\documentclass[a4paper,11pt]{article}

\usepackage{geometry}
\usepackage{parskip}
\usepackage[hidelinks]{hyperref}
\usepackage{amsmath,amssymb}
\usepackage{subcaption}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,cipher}

\colorlet{diffcolor}{red}

\begin{document}

\pagestyle{empty}
% PRESENT CHARACTERISTIC
\clearpage
\begin{figure}
  \centering
  \newcommand{\activeInput}[1]{
    \draw[activebit,->] (i0-#1) -- (i0-#1|-S1-0.north);
  }
  \newcommand{\activeLinear}[2]{
    % #1 = round, #2 = bit position at sboxout
    \pgfmathsetmacro{\newPos}{int(mod(\nbits/4*#2,\lastBit)+div(#2,\lastBit)*\lastBit)}
    \draw[activebit,->] (i0-#2|-S#1-0.south) -- +(0,-.5em) -- (i#1-\newPos) -- (xor#1-\newPos.south);
    \pgfmathsetmacro{\actSbx}{int(div(#2,4))}
    \activeSbox{#1}{\actSbx}
  }
  \newcommand{\activeSbox}[2]{
    % #1 = round, #2 = sbox position
    \node[box,diffcolor] (S#1-#2) at (#2*3em+1.1em,7em-#1*9em) {\color{white}$\mathcal{S}$};
  }
  \newcommand{\sboxLabel}[4]{
    % #1 = round, #2 = sbox position, #3 input label, #4 = output label
    %\node[box,diffcolor] (S#1-#2) at (#2*3em+1.1em,7em-#1*9em) {};
    \node[below=-1mm of S#1-#2.north west] {\hspace{0.5em}{\scriptsize\color{white}\texttt{#3}}};
    \node[above=-1mm of S#1-#2.south west] {\hspace{0.5em}{\scriptsize\color{white}\texttt{#4}}};
  }

  \begin{tikzpicture}[scale=.625,
                      rounded corners,
                      >=latex,
                      box/.append style={minimum size=.625cm},
                      activebit/.style={diffcolor,very thick}
    ]
    """.strip("\n")

_NUM_ROUNDS_MACRO = r"""\pgfmathsetmacro{\nrounds}""".strip("\n")

_DRAWING_LATEX = r"""
    \pgfmathsetmacro{\nbits}{64}  % set bitsize of state
    \pgfmathsetmacro{\lastBit}{int(\nbits-1)}
    \pgfmathsetmacro{\secondlastBit}{int(\nbits-2)}
    \pgfmathsetmacro{\lastSbox}{int(\nbits/4-1)}
    \foreach \r in {0,...,\nrounds} {
      \foreach \z in {0,...,\lastBit} {
        \node[xor, scale=0.6] (xor\r-\z) at (\z*0.75em,-\r*9em) {};
        \coordinate[above = 0.5em of xor\r-\z] (i\r-\z) ;
      }
      %\node[left = 0em of xor\r-0] (K\r) {$RK_{\r}$};
    }
    \foreach \z [evaluate=\z as \zz using {int(mod(16*\z,\lastBit))}] in {0,...,\lastBit} {
      \draw[->] (i0-\z) -- (xor0-\z);
    }
    \foreach \r [evaluate=\r as \rr using {int(\r-1)}] in {1,...,\nrounds} {
      \foreach \z in {0,...,\lastSbox} {
        \node[box] (S\r-\z) at (\z*3em+1.1em,7em-\r*9em) {$\mathcal{S}$};
      }
      \foreach \z [evaluate=\z as \zz using {int(mod(16*\z,\lastBit))}] in {0,...,\secondlastBit} {
        \draw[->] (xor\rr-\z|-S\r-0.south) -- +(0,-.5em) -- (i\r-\zz) -- (xor\r-\zz);
        \draw     (xor\rr-\z) -- (xor\rr-\z|-S\r-0.north);
      }
      \draw[->] (xor\rr-\lastBit|-S\r-0.south) -- +(0,-.5em) -- (i\r-\lastBit) -- (xor\r-\lastBit);
      \draw     (xor\rr-\lastBit) -- (xor\rr-\lastBit|-S\r-0.north);
    }

""".strip("\n")

_DOCUMENT_END = r"""
  \end{tikzpicture}
\end{figure}
\end{document}
""".strip("\n")


import numpy as np

class PresentCharacteristic(DifferentialCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint8]] = PRESENT_DDT
    sbox_count = 16

    def __init__(self, sbox_in: np.ndarray, sbox_out: np.ndarray, file_path: Path|None = None):
        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in and sbox_out must have the same shape')
        if len(sbox_in.shape) != 2 or len(sbox_out.shape) != 2:
            raise ValueError('sbox_in and sbox_out must have 2 dimensions')
        if sbox_in.shape[1] != 16 or sbox_out.shape[1] != 16:
            raise ValueError('sbox_in and sbox_out must have 16 s-boxes')

        num_rounds = sbox_in.shape[0]
        permuted = bit_perm(sbox_out)

        for i in range(1, num_rounds):
            lin_input = sbox_out[i - 1]
            lin_output = sbox_in[i]
            permuted = bit_perm(lin_input)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')

        super().__init__(sbox_in, sbox_out, file_path=file_path)


    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        with np.load(characteristic_path) as f:
            sbox_in = f['sbox_in']
            sbox_out = f['sbox_out']
        sbox_in = np.array(sbox_in, dtype=np.uint8)
        sbox_out = np.array(sbox_out, dtype=np.uint8)

        return cls(sbox_in, sbox_out, file_path=characteristic_path)

    @classmethod
    def load_from_model(cls, model):
        sbox_in = model.sbox_in[:-1]
        sbox_out = model.sbox_out
        return cls(sbox_in, sbox_out, file_path=None)


    def tikzify(self)  -> str:
        result = StringIO()
        print(_PREAMBLE, file=result)
        print(_NUM_ROUNDS_MACRO + "{" + str(self.num_rounds) + "}", file=result)
        print(_DRAWING_LATEX, file=result)

        # round 0:
        for nibble_idx in range(self.sbox_in.shape[1]):
            for bit_idx in range(4):
                if self.sbox_in[0, nibble_idx] & (1 << bit_idx):
                    print(f"    \\activeInput{{{4 * nibble_idx + bit_idx}}}", file=result)

        # remaining rounds
        for round in range(self.num_rounds):
            for nibble_idx in range(self.sbox_in.shape[1]):  # num_sboxes
                for bit_idx in range(4):
                    if self.sbox_out[round, nibble_idx] & (1 << bit_idx):
                        print(f"    \\activeLinear{{{round + 1}}}{{{4 * nibble_idx + bit_idx}}}", file=result)

        print(_DOCUMENT_END, file=result)
        return result.getvalue()

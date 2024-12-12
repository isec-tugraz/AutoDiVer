from __future__ import annotations

from typing import Any, TYPE_CHECKING
from io import StringIO

import numpy as np

from ..cipher_model import DifferentialCharacteristic
from .gift_util import bit_perm, P64, P128, DDT as GIFT_DDT

if TYPE_CHECKING:
    from pathlib import Path

_PREAMBLE = r"""
\documentclass{standalone}
\usepackage{tikz}
\usepackage{gift}
\usepackage{tugcolors}
\begin{document}

""".strip("\n")

_DOCUMENT_END = r"""
\end{document}
""".strip("\n")



class _GiftCharacteristic(DifferentialCharacteristic):
    ddt: np.ndarray[Any, np.dtype[np.uint8]] = GIFT_DDT

    num_sboxes: int
    block_size: int
    permutation: np.ndarray

    _round_keys: np.ndarray[Any, np.dtype[np.int32]]

    @classmethod
    def verify(cls, sbox_in: np.ndarray, sbox_out: np.ndarray) -> None:
        if len(sbox_in.shape) != 2:
            raise ValueError(f'sbox_in must have 2 dimensions (round, sbox), sbxo_in.shape = {sbox_in.shape}')

        if sbox_in.shape != sbox_out.shape:
            raise ValueError('sbox_in.shape must equal sbox_out.shape')

        if sbox_in.shape[1] != cls.num_sboxes:
            raise ValueError(f'sbox_in.shape[1] must be {cls.num_sboxes}')

        num_rounds = sbox_in.shape[0]

        for i in range(1, num_rounds):
            lin_input = sbox_out[i - 1]
            lin_output = sbox_in[i]
            permuted = bit_perm(lin_input, cls.permutation)
            if not np.all(permuted == lin_output):
                raise ValueError(f'linear layer condition violated at sbox_out[{i - 1}] -> sbox_in[{i}]')

    @classmethod
    def load(cls, characteristic_path: Path) -> DifferentialCharacteristic:
        if characteristic_path.suffix == '.txt':
            return cls.load_txt(characteristic_path)
        elif characteristic_path.suffix == '.npz':
            return cls.load_npz(characteristic_path)
        else:
            raise ValueError(f'unsupported file type {characteristic_path.suffix}')

    def tikzify(self) -> str:
        result = StringIO()

        inv_perm = np.zeros_like(self.permutation)
        inv_perm[self.permutation] = np.arange(len(self.permutation), dtype=self.permutation.dtype)


        print(_PREAMBLE, file=result)
        print(r"""
  \begin{tikzpicture}[gift, xscale=.8, sbox/.append style={minimum size=.4cm}, op/.append style={scale=.8}]
              """.strip("\n").rstrip(), file=result)

        cmd = "smallgiftfalse" if self.block_size == 128 else "smallgifttrue"
        print(f"      \\{cmd}", file=result)
        print(f"      \\giftinit[i]", file=result)
        print(f"      \\spnlinktrue", file=result)

        num_rounds = len(self.sbox_in)
        for rnd in range(num_rounds):
            print(f"      \\giftround", file=result)

            active_sboxes, = np.nonzero(self.sbox_in[rnd])
            active_input_bits = []
            active_output_bits = []
            for nibble_idx in range(self.num_sboxes):
                for bit_idx in range(4):
                    if self.sbox_in[rnd, nibble_idx] & (1 << bit_idx):
                        active_input_bits.append(4 * nibble_idx + bit_idx)
                    if self.sbox_out[rnd, nibble_idx] & (1 << bit_idx):
                        active_output_bits.append(4 * nibble_idx + bit_idx)

            active_sboxes_str = ",".join(str(x) for x in active_sboxes)
            active_input_bits_str = ",".join(str(x) for x in active_input_bits)
            active_bit_transitions = ",".join(f"{x}/{inv_perm[x]}" for x in active_output_bits)
            print(f"      \\giftmarkbits[tugblue]{{{active_input_bits_str}}}{{{active_sboxes_str}}}{{{active_bit_transitions}}}", file=result)

        print(f"      \\giftfini", file=result)
        print(r"      \foreach \i in {0,4,...,\bits} { \draw (b\i|-here) node[below,gray,inner sep=0pt,font=\tiny] {\i}; }", file=result)



        print(r"""
  \end{tikzpicture}
        """.strip("\n").rstrip(), file=result)

        print(_DOCUMENT_END, file=result)
        return result.getvalue()


class Gift64Characteristic(_GiftCharacteristic):
    num_sboxes = 16
    block_size = 64
    permutation = P64

class Gift128Characteristic(_GiftCharacteristic):
    num_sboxes = 32
    block_size = 128
    permutation = P128

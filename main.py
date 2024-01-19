#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
from cipher_model import SboxCipher, DifferentialCharacteristic
from gift64.gift64 import Gift64
def main():
    ciphers: dict[str, type[SboxCipher]] = {
        "gift64": Gift64,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cipher', choices=ciphers.keys())
    parser.add_argument('trail', help='Text file containing the sbox input and output differences.\n'\
                                      'Input and output differences are listed on separate lines.')
    parser.add_argument('--epsilon', type=float, default=0.8)
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--cnf', type=str, help="file to save CNF in DIMACS format")
    args = parser.parse_args()
    trail = []
    with open(args.trail, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            assert len(line) == 16
            line_deltas = [int(l, 16) for l in line[::-1]]
            trail.append(line_deltas)
    trail = np.array(trail)
    if len(trail) % 2 != 0:
        print(f'expected an even number of differences in {args.trail!r}')
        raise SystemExit(1)
    sbox_in = trail[0::2]
    sbox_out = trail[1::2]
    char = DifferentialCharacteristic(sbox_in, sbox_out)
    Cipher = ciphers[args.cipher]
    cipher = Cipher(char)
    ddt_prob = char.log2_ddt_probability(Cipher.ddt)
    print(f"ddt probability: 2**{ddt_prob:.1f}")
    if args.cnf:
        with open(args.cnf, 'w') as f:
            f.write(cipher.cnf.to_dimacs())
        print(f"wrote cnf to {args.cnf}")
    cipher.count_key_space(args.epsilon, args.delta, verbosity=0)
    # for _ in range(10):
    #     gift.count_probability_for_random_key(verbosity=0)
    # gift.count_probability(args.epsilon, args.delta, verbosity=2)
    # from IPython import embed; embed()
if __name__ == "__main__":
    raise SystemExit(main())
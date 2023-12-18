#!/usr/bin/env python3
import numpy as np
from sat_toolkit.formula import CNF, Truthtable
from pyapproxmc import Counter
sbox = np.array((1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14), dtype=np.uint8)
def get_sbox_cnf(sbox):
    """
    reutrn
    """
    sbox_size = len(sbox)
    lut = np.zeros((sbox_size, sbox_size), dtype=np.uint8)
    lut[(range(sbox_size), sbox)] = 1
    tt = Truthtable.from_lut(lut.reshape(-1))
    return tt.to_cnf()
if __name__ == '__main__':
    cnf = get_sbox_cnf(sbox)
    counter = Counter()
    for clause in cnf:
        counter.add_clause(clause)
    print(counter.count())
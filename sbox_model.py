from sat_toolkit.formula import CNF, Truthtable
import numpy as np
"""
Generaic model for valid input and output values
Inputs:
    sbox: as a list, e.g., GIFT = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
    m: number of input bits
    n: number of ouput bits
Output:
    output is a reduced CNF of m+n variables y3y2y1y0||x3x2x1x0 such that S(x) = y
"""
def sbox_propagation_to_cnf(sbox, m, n):
    table = np.zeros((2**m)*(2**n), int)
    for x in range(0, 2**m):
        y = sbox[x]
        a = (y << m) | x
        table[a] = 1
    tt = Truthtable.from_lut(table)
    cnf = tt.to_cnf()
    return cnf
"""
Replace the generaic SBOX model with a specific [input, output] variables
Inputs:
    cnf: Generaic CNF
    vars: [[input_variables, output_variables]
Output:
    output is a reduced CNF of m+n variables x||y such that S(x) = y
"""
def sbox_model(cnf, variables):
    varrs = [0] + variables
    varrs = np.array(varrs, np.int32)
    cnf_new = cnf.translate(varrs)
    return cnf_new
if __name__ == "__main__":
    sbox = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
    m = 4
    n = 4
    cnf = sbox_propagation_to_cnf(sbox, m, n)
    print(cnf)
    variables = [10 + i for i in range(m+n)]
    cnf_new = sbox_model(cnf, variables)
    print(cnf)
    print(cnf_new)
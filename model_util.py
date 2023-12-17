from sat_toolkit.formula import CNF, Truthtable
import numpy as np
"""
Inputs:
    sbox: as a list, e.g., GIFT = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
    m: number of input bits
    n: number of ouput bits
    inputDiff : hex value of input difference
    outputDiff: hex value of output difference
Output:
    output is a reduced CNF of m+n variables y3y2y1y0||x3x2x1x0 such that S(x) = y
"""
def sboxModel(sbox, m, n, inputDiff, outputDiff, variables):
    table = np.zeros((2**m)*(2**n), int)
    count = 0
    for x in range(0, 2**m):
        y = sbox[x]
        y_prime = sbox[x ^ inputDiff]
        if ((y ^ y_prime) == outputDiff):
            a = (y << m) | x
            table[a] = 1
            count += 1
    tt = Truthtable.from_lut(table)
    cnf = tt.to_cnf()
    # print(cnf)
    # print(f'{count = }')
    cnf_new = cnf.translate(np.array(variables, np.int32))
    return cnf_new
"""
Inputs:
    variables:
Output:
    output is a reduced CNF of 1+2 variables y||x1x0 such that y=x0+x1
0| 0 0 => 0
1| 0 1 => 5
1| 1 0 => 6
0| 1 1 => 3
"""
def xorModel(variables):
    table = np.zeros(8, int)
    table[[0, 5, 6, 3]] = 1
    tt = Truthtable.from_lut(table)
    cnf = tt.to_cnf()
    cnf_new = cnf.translate(np.array(variables, np.int32))
    return cnf_new
"""
Inputs:
    variables:
Output:
    output is a reduced CNF of 1+1 variables y||x such that y=x
0 0 => 0
1 1 => 3
"""
def eqModel(variables):
    table = np.zeros(4, int)
    table[[0, 3]] = 1
    tt = Truthtable.from_lut(table)
    cnf = tt.to_cnf()
    cnf_new = cnf.translate(np.array(variables, np.int32))
    return cnf_new
if __name__ == "__main__":
    sbox = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
    m = 4
    n = 4
    inD = 0x4
    outD = 0x7
    variables = [0] + [10 + i for i in range(m+n)]
    cnf = sboxModel(sbox, m, n, inD, outD, variables)
    print(cnf)
    variables = [0] + [5,6,7]
    cnf = xorModel(variables)
    print(cnf)
    variables = [0] + [1,2]
    cnf = eqModel(variables)
    print(cnf)
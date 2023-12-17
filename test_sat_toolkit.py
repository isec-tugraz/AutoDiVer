from sat_toolkit.formula import CNF, Truthtable
import numpy as np
table = np.zeros(16, int)
table[[0, 6, 9, 15]] = 1
tt = Truthtable.from_lut(table)
cnf = tt.to_cnf()
print(cnf)
tt = cnf.translate(np.array([0, 8, 9, 10, 11], np.int32))
print(tt)
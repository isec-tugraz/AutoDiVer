from __future__ import annotations

import numpy as np
import os

from sat_toolkit.formula import XorCNF, CNF
from autodiver.util import IndexSet
from autodiver.arx_util import  modular_addition_probability, _hamming_weight
from pysat.card import CardEnc, IDPool



# exploratory: find DDT for 4 bit adder (to get an idea)
bitwidth = 32
# ddt = np.zeros((pow(2, bitwidth), pow(2, bitwidth), pow(2, bitwidth)))
# for a in range(pow(2, bitwidth)):
#     for b in range(pow(2, bitwidth)):
#         for y in range(pow(2, bitwidth)):
#             ddt[a,b,y] = modular_addition_probability(np.array([a]), np.array([b]), np.array([y]), bitwidth)

# for table in ddt:
#     print(table)

###################################################################################

cnf = XorCNF()
set = IndexSet()

set.add_index_array("a", (bitwidth,))
set.add_index_array("b", (bitwidth,))
set.add_index_array("y", (bitwidth,))

set.add_index_array("aux1", (bitwidth - 1,))
set.add_index_array("aux2", (bitwidth - 1,))
set.add_index_array("aux3", (bitwidth - 1,))
set.add_index_array("weights", (bitwidth - 1,))

##################################################################################
# encode validity condition and weights

# index 0 : a[-1] = 0
cnf += XorCNF.create_xor([set.a[0]], [set.b[0]], [set.y[0]]) # default rhs is 0 here


cnf += XorCNF.create_xor(set.aux1, -set.a[:-1], set.b[:-1])
cnf += XorCNF.create_xor(set.aux2, -set.a[:-1], set.y[:-1])
cnf += XorCNF.create_xor(set.aux3, set.a[1:], set.b[1:], set.y[1:], set.b[:-1])

reg_CNF = CNF()

for i in range(bitwidth-1):
    #cnf += XorCNF.create_xor(set.aux1[i-1], -set.a[i-1], set.b[i-1])
    #cnf += XorCNF.create_xor(set.aux2[i-1], -set.a[i-1], set.y[i-1])
    #cnf += XorCNF.create_xor(set.aux3[i-1], set.a[i], set.b[i], set.y[i], set.b[i-1])
    reg_CNF += [-set.aux1[i], -set.aux2[i], -set.aux3[i], 0]

    # weight encoding:
    reg_CNF += [set.aux1[i], set.weights[i], 0]
    reg_CNF += [set.aux2[i], set.weights[i], 0]
    reg_CNF += [-set.weights[i], -set.aux1[i], -set.aux2[i], 0]

cnf += reg_CNF

##########################################################################
# cardinality encoding:

bound = 0
vpool = IDPool(start_from=set.numvars + 1)
cardinality_encoding = CardEnc.atmost(lits=set.weights.flatten().tolist(), vpool=vpool, bound=bound).clauses

cardinality_encoding_cnf = CNF()

for clause in cardinality_encoding:
    cardinality_encoding_cnf += clause + [0]

cnf += cardinality_encoding_cnf

print(cnf.to_dimacs())
seed = int.from_bytes(os.urandom(4), 'little')
args = ['cryptominisat5', f'--random={seed}', '--polar=rnd']
is_sat, raw_model = cnf.solve_dimacs(args)
print(is_sat, raw_model)
model = set.get_model(raw_model)
print(vars(model))

#prob = ddt[model.a, model.b, model.y]
print(f"real probability: {modular_addition_probability(np.array([model.a]), np.array([model.b]), np.array([model.y]), bitwidth)}")
print(f"modeled probability: {1.0 / (1 << _hamming_weight(model.weights))}")






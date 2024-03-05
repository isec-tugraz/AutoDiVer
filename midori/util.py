from typing import Literal
from sat_toolkit.formula import XorCNF
import numpy as np
DDT  = np.array([[16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 2, 4, 0, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0],
                 [0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, 4, 2, 2, 2, 0, 0, 0, 2, 0, 2],
                 [0, 2, 4, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
                 [0, 2, 0, 0, 2, 0, 0, 4, 0, 2, 4, 0, 2, 0, 0, 0],
                 [0, 2, 0, 4, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 2],
                 [0, 0, 0, 2, 0, 4, 2, 0, 0, 0, 0, 2, 0, 4, 2, 0],
                 [0, 2, 0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0, 2, 0],
                 [0, 0, 4, 2, 0, 2, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 4],
                 [0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 4, 0, 2, 0, 2],
                 [0, 0, 4, 0, 0, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0],
                 [0, 0, 0, 2, 0, 0, 2, 4, 0, 0, 4, 2, 0, 0, 2, 0],
                 [0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 4, 2],
                 [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 4, 2, 0, 0, 2, 4]],
                 dtype=np.uint8)
RC = np.array([[0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,1],
               [0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0],
               [1,0,1,0,0,1,0,0,0,0,1,1,0,1,0,1],
               [0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,1],
               [0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1],
               [1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0],
               [0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0],
               [0,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0],
               [1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1],
               [0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0],
               [0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,1],
               [0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0],
               [0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,0],
               [1,1,1,1,1,0,0,0,1,1,0,0,1,0,1,0],
               [1,1,0,1,1,1,1,1,1,0,0,1,0,0,0,0]],
               dtype=np.uint8)
# 0, 10, 5, 15,
# 14, 4, 11, 1,
# 9, 3, 12, 6,
# 7, 13, 2, 8
sr_mapping = np.array([0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8])
mixing_mat = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],])
def do_shift_rows(state):
    return state[sr_mapping]
def model_mix_cols(A, B):
    mc_cnf = XorCNF()
    for c in range(4):
        colA = A[(4*c):(4*c)+4]
        colB = B[(4*c):(4*c)+4]
        for r in range(4):
            colA_red = colA[mixing_mat[r] != 0, :]
            print(f'{colB[r]}', "===>", f'{colA_red}')
            mc_cnf += XorCNF.create_xor(colB[r], *colA_red)
    return mc_cnf
# def do_mix_cols(state):
#     state_2d = state.reshape(4, 4)
#     print(f'{state_2d = }')
#     result = np.zeros_like(state_2d)
#     for col in range(4):
#         result[col, :] = np.bitwise_xor.reduce(mixing_mat * state_2d[col, :], axis=-1)
#     return result
def do_linear_layer(state):
    state = do_shift_rows(state)
    state = do_mix_cols(state)
    return state.flatten()
if __name__ == "__main__":
    A = np.array([[4*j + i for i in range(4)] for j in range(16)])
    B = do_shift_rows(A)
    print(f'{A=}')
    print(f'{B=}')
    A = np.array([[4*j + i+1 for i in range(4)] for j in range(16)])
    B = np.array([[4*j + i+1 for i in range(4)] for j in range(16)])
    print(f'{A=}')
    print(f'{B=}')
    model_mix_cols(A, B)
    print(f'{A=}')
    print(f'{B=}')
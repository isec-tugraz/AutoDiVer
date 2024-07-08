"""
In the model if a byte is represented by 0,1,2,3,4,5,6,7 then
the byte representation will be =>> 3210 7654 <<=
"""


import numpy as np
# def Perm(state, perm):
#     temp = state[:]
#     for i in range(len(state)):
#         state[i] = temp[perm[i]]
#     return state
# def PermInv(state, perm):
#     temp = state[:]
#     for i in range(len(state)):
#         state[perm[i]] = temp[i]
#     return state
# def pre_permute_byte(a, i):
#     perm  = np.array((3,2,1,0, 7,6,5,4))
#     perm0 = np.array((4,1,6,3, 0,5,2,7))
#     perm1 = np.array((1,6,7,0, 5,2,3,4))
#     perm2 = np.array((2,3,4,1, 6,7,0,5))
#     perm3 = np.array((7,4,1,2, 3,0,5,6))
#     b = Perm(a, perm)
#     if i == 0:
#         b = Perm(b, perm0)
#     if i == 1:
#         b = Perm(b, perm1)
#     if i == 2:
#         b = Perm(b, perm2)
#     if i == 3:
#         b = Perm(b, perm3)
#     return b
# def post_permute_byte(a, i):
#     perm  = np.array((3,2,1,0, 7,6,5,4))
#     perm0 = np.array((4,1,6,3, 0,5,2,7))
#     perm1 = np.array((1,6,7,0, 5,2,3,4))
#     perm2 = np.array((2,3,4,1, 6,7,0,5))
#     perm3 = np.array((7,4,1,2, 3,0,5,6))
#     b = a[perm]
#     if i == 0:
#         b = PermInv(b, perm0)
#     if i == 1:
#         b = PermInv(b, perm1)
#     if i == 2:
#         b = PermInv(b, perm2)
#     if i == 3:
#         b = PermInv(b, perm3)
#     # b = PermInv(b, perm)
#     return b
# def post_permutation():
#     A = [[8*j + i for i in range(8)] for j in range(16)]
#     B = []
#     for i in range(16):
#         a = np.asarray(A[i])
#         a = post_permute_byte(a, i%4)
#         B.append(a)
#     B = np.asarray(B).flatten()
#     print(B)
#     return B


def permute_byte(a, i):
    perm  = np.array((3,2,1,0, 7,6,5,4))
    perm0 = np.array((4,1,6,3, 0,5,2,7))
    perm1 = np.array((1,6,7,0, 5,2,3,4))
    perm2 = np.array((2,3,4,1, 6,7,0,5))
    perm3 = np.array((7,4,1,2, 3,0,5,6))
    b = a[perm]
    if i == 0:
        b = b[perm0]
    elif i == 1:
        b = b[perm1]
    elif i == 2:
        b = b[perm2]
    elif i == 3:
        b = b[perm3]
    else:
        raise ValueError("Invalid i: {i}")
    b = b[perm]
    return b


def permutation():
    A = [[8*j + i for i in range(8)] for j in range(16)]
    B = []
    for i in range(16):
        a = np.asarray(A[i])
        a = permute_byte(a, i%4)
        B.append(a)
    B = np.asarray(B).flatten()
    # print(B)
    return B
# permutation()
# post_permutation()

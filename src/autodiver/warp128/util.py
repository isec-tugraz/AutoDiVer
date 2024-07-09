import numpy as np
DDT  = np.array([[16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 4, 0, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0], [0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 2, 0, 4, 2, 2, 2, 0, 0, 0, 2, 0, 2], [0, 2, 4, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0], [0, 2, 0, 0, 2, 0, 0, 4, 0, 2, 4, 0, 2, 0, 0, 0], [0, 2, 0, 4, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 2], [0, 0, 0, 2, 0, 4, 2, 0, 0, 0, 0, 2, 0, 4, 2, 0], [0, 2, 0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0, 2, 0], [0, 0, 4, 2, 0, 2, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 4], [0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 4, 0, 2, 0, 2], [0, 0, 4, 0, 0, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0], [0, 0, 0, 2, 0, 0, 2, 4, 0, 0, 4, 2, 0, 0, 2, 0], [0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 4, 2], [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 4, 2, 0, 0, 2, 4]], dtype=np.uint8)
RC = np.array([[0x0, 0x0, 0x1, 0x3, 0x7, 0xf, 0xf, 0xf, 0xe, 0xd, 0xa, 0x5, 0xa, 0x5, 0xb, 0x6, 0xc, 0x9, 0x3, 0x6, 0xd, 0xb, 0x7, 0xe, 0xd, 0xb, 0x6, 0xd, 0xa, 0x4, 0x9, 0x2, 0x4, 0x9, 0x3, 0x7, 0xe, 0xc, 0x8, 0x1, 0x2], [0x4, 0xc, 0xc, 0xc, 0xc, 0xc, 0x8, 0x4, 0x8, 0x4, 0x8, 0x4, 0xc, 0x8, 0x0, 0x4, 0xc, 0x8, 0x4, 0xc, 0xc, 0x8, 0x4, 0xc, 0x8, 0x4, 0x8, 0x0, 0x4, 0x8, 0x0, 0x4, 0xc, 0xc, 0x8, 0x0, 0x0, 0x4, 0x8, 0x4, 0xc]], dtype=np.uint8);
perm = [31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10, 15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26]


def perm_nibble_inv(temp):
    state = [0 for i in range(32)]
    for j in range(32):
        state[j] = temp[perm[j]]
    return np.asarray(state, np.uint32)


def perm_nibble(temp):
    state = [0 for i in range(32)]
    for j in range(32):
        state[perm[j]] = temp[j]
    return np.asarray(state, np.uint32)


def perm_nibble_16(state):
    state1 = [0 for i in range(32)]
    for i in range(16):
        state1[2*i] = state[i]
    temp = [0 for i in range(32)]
    for i in range(32):
        temp[perm[i]] = state1[i]
    state2 = [0 for i in range(16)]
    for i in range(16):
        state2[i] = temp[2*i + 1]
    return np.asarray(state2, np.uint32)


def perm_nibble_16_inv(state):
    state1 = [0 for i in range(32)]
    for i in range(16):
        state1[2*i] = state[i]
    temp = [0 for i in range(32)]
    for i in range(32):
        temp[i] = state1[perm[i]]
    state2 = [0 for i in range(16)]
    for i in range(16):
        state2[i] = temp[2*i + 1]
    return np.asarray(state2, np.uint32)


def get_round_val(a, b):
    v = np.empty(32, dtype=np.uint8)
    for i in range(16):
        v[2*i] = a[i]
        v[2*i+1] = b[i]
    return v


def get_round_in_out(rounds, model):
    rounds_in = np.empty((rounds, 32), dtype=np.uint8)
    rounds_out = np.empty((rounds, 32), dtype=np.uint8)
    for i in range(0, rounds-1):
        rounds_out[i] = get_round_val(model.sbox_in[i], perm_nibble_16_inv(model.sbox_in[i+1]))
    #no permutation at the end
    rounds_out[rounds-1] = get_round_val(model.sbox_in[rounds-1], model.Y)
    return rounds_out

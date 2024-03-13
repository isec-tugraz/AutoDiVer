import numpy as np
DDT  = np.array([[16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 4, 0, 2, 0, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0], [0, 4, 0, 2, 4, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0], [0, 0, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 0], [0, 2, 4, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 4, 0, 2], [0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 4], [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0], [0, 2, 0, 0, 0, 0, 2, 4, 0, 0, 0, 2, 0, 2, 2, 2], [0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 4, 2, 4, 0, 0, 0], [0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 0, 2, 4, 0], [0, 0, 0, 2, 2, 0, 0, 0, 4, 2, 2, 2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2], [0, 2, 0, 2, 0, 2, 2, 0, 4, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 2, 4, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 2], [0, 0, 2, 0, 0, 2, 2, 2, 0, 4, 0, 2, 0, 2, 0, 0], [0, 0, 0, 0, 2, 4, 0, 2, 0, 0, 2, 2, 0, 2, 0, 2]], dtype=np.uint8)
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
               [1,1,0,1,1,1,1,1,1,0,0,1,0,0,0,0],
               [0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1],
               [0,0,0,1,1,1,0,0,0,0,1,0,0,1,0,0],
               [0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0],
               [0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,0]],
               dtype=np.uint8)
def nibble_to_byte(state):
    out_state = []
    for i in range(16):
        a = (state[2*i] << 4) | state[(2*i) + 1]
        out_state.append(a)
    out_state = np.asarray(out_state, dtype = np.uint8)
    return out_state
def byte_to_nibble(state):
    out_state = [0 for _ in range(32)]
    for i in range(16):
        a = state[i] & 0x0F
        out_state[2*i] = a
        a = (state[i] >> 4) & 0x0F
        out_state[2*i + 1] = a
    out_state = np.asarray(out_state, dtype = np.uint8)
    return out_state
# 0, 10, 5, 15,
# 14, 4, 11, 1,
# 9, 3, 12, 6,
# 7, 13, 2, 8
sr_mapping = np.array([0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8])
sri_mapping = np.array([0, 7, 14, 9, 5, 2, 11, 12, 15, 8, 1, 6, 10, 13, 4, 3])
mixing_mat = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],])
def do_shift_rows(state):
    return state[sr_mapping]
def do_shift_rows_inv(state):
    return state[sri_mapping]
def do_mix_columns(state):
    out_state = [0 for i in range(16)]
    for c in range(4):
        colA = state[(4*c):(4*c)+4]
        for r in range(4):
            colA_red = colA[mixing_mat[r] != 0]
            out_state[4*c + r] = np.bitwise_xor.reduce(colA_red)
    out_state = np.asarray(out_state, dtype = np.uint8)
    return out_state
def do_linear_layer(state):
    state = nibble_to_byte(state)
    state = do_shift_rows(state)
    state = do_mix_columns(state)
    state = byte_to_nibble(state)
    return state.flatten()
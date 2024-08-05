import numpy as np
from typing import Any

# the loadstate functionality of the c implementation is not part of the SAT model -> pre- and post processing needed
def load_state(plaintext: np.ndarray[Any, np.dtype[np.int8]], state_size) -> np.ndarray[Any, np.uint32]:
    state = np.ndarray(state_size, dtype=np.int32)
    for i in range(state_size):
        state[i] = plaintext[4 * i + 0]
        state[i] = (state[i] << 8) | plaintext[4 * i + 1]
        state[i] = (state[i] << 8) | plaintext[4 * i + 2]
        state[i] = (state[i] << 8) | plaintext[4 * i + 3]
    return state


def unload_state(state: np.ndarray[Any, np.dtype[np.uint32]], state_size) -> np.ndarray[Any, np.uint8]:
    ciphertext = np.ndarray(state_size*4, dtype=np.uint8)
    for i in range(state_size):
        ciphertext[4 * i + 0] = 0b11111111 & (state[i] >> 24)
        ciphertext[4 * i + 1] = 0b11111111 & (state[i] >> 16)
        ciphertext[4 * i + 2] = 0b11111111 & (state[i] >> 8)
        ciphertext[4 * i + 3] = 0b11111111 & (state[i] >> 0)
    return ciphertext


def pyjamask_mat_mult(mat_col: np.uint32, vec: np.uint32) -> np.uint32:
    mask = np.uint32(0)
    res = np.uint32(0)

    for i in range(31, -1, -1):
        mask = -np.int32((vec >> i) & 1)
        res = res ^ (mask & mat_col)
        mat_col = (mat_col >> 1) | (mat_col << 31)

    return res

def pyjamask_mix_rows_96(state: np.ndarray[Any, np.uint32]) -> np.ndarray[Any, np.uint32]:

    assert state.shape[0] == 3
    state_out = state.copy()
    state_out[0] = pyjamask_mat_mult(np.uint32(0xa3861085), state[0])
    state_out[1] = pyjamask_mat_mult(np.uint32(0x63417021), state[1])
    state_out[2] = pyjamask_mat_mult(np.uint32(0x692cf280), state[2])

    return state_out

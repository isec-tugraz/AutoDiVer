import numpy as np
from typing import Any

SBOX = np.array(bytearray.fromhex('08000903381029130c0d0407300120231a1218323e162c361c1d14373405242702060b0f331721150a1b0e1f3111253522262a2e3a1e283c2b3b2f3f39192d3d'))
RC = np.array(bytearray.fromhex(
    '09033d2a22081623020d0c130618282e00370133110a10090e02082927330710'
    '02023b3a260e310e1b0825050a0207260e0d00131d3b3914192c3c343a10312c'
    '300a30292d3c253c140d343f210d16352d141c0905390816351d26091e1f2c1b'
    '3413040b2929231f2d1a302f3f170b1b34012b1f2d3b23212b3e352a09273a16'
    '2e273210111f042c1f392424281925072c39052c3d3020013c2e0a05232f3016'
    '183624203607051713262624160f3a233d090c3d1f2036151d083d32232b1918')).reshape(-1, 32)

def do_shift_cols(state: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """
    Perform shift columns. Each row is layed out in little endian order.
    I.e., the bit at index[0] is the LSB which is shifted by 5 places.
    """
    if state.shape != (32, 6):
        raise ValueError('state must have shape (32, 6)')

    result = np.zeros_like(state)
    for i in range(6):
        result[:, 5 - i] = np.roll(state[:, 5 - i], -i)
    return result

def do_mix_cols(state: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """ Perform mix columns on the state of bits. """
    if state.shape != (32, 6):
        raise ValueError('state must have shape (32, 6)')
    alphas = (0, 1, 5, 9, 15, 21, 26)

    mixed = np.zeros_like(state)
    for alpha in alphas:
        mixed ^= np.roll(state, -alpha, axis=0)
    return mixed


def update_key(key):
    keyr = np.zeros((32, 6), dtype = np.uint8)
    for i in range(32):
        keyr[i] = key[i][::-1]
    k = keyr.flatten()

    key1 = key.copy().flatten()
    for i in range(192):
        key1[i] = k[(7*i + 1)%192]
    key1 = key1.reshape(32, 6)
    # print(key1)

    key2 = key1.copy()
    for i in range(32):
        key2[i] = key1[i][::-1]
    return key2

def Add(A, B):
    # print(f'{A = }')
    # print(f'{B = }')
    assert A.shape == B.shape
    state = []
    for i in range(len(A)):
        s = A[i] ^ B[i]
        state.append(s)
    return np.asarray(state, np.uint8)

def convert_statebool_to_statechar(input_state):
    output_state = [0 for i in range(32)]
    for i in range(32):
        for j in range(6):
            output_state[i] <<= 1
            output_state[i] ^= input_state[6 * i + j]
    return output_state

def prepare_round_keys(key):
    NR = 7
    round_keys = [[0 for i in range(32)] for j in range(NR)]
    temp_round_key_state = [[0 for _ in range(192)] for _ in range(2)]
    # print(f'{temp_round_key_state = }')

    #convert 32 nibble key to 192 bits
    for i in range(32):
        for j in range(6):
            temp_round_key_state[0][6 * i + j] = (key[i] >> (5 - j)) & 1

    round_keys[0] = convert_statebool_to_statechar(temp_round_key_state[0])

    for r in range(1, NR):
        ind_new = (r % 2)
        ind_old = ~ind_new
        for i in range(192):
            temp_round_key_state[ind_new][i] = temp_round_key_state[ind_old][(7 * i + 1) % 192]
        round_keys[r] = convert_statebool_to_statechar(temp_round_key_state[ind_new])
    return np.asarray(round_keys, np.uint8)


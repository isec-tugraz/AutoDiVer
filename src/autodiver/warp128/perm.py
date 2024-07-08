perm = [31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10, 15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26]


def perm_nibble_16(state):
    state1 = [0 for i in range(32)]
    for i in range(16):
        state1[2*i] = state[i]
    temp = [0 for i in range(32)]
    for i in range(32):
        temp[perm[i]] = state1[i]
    print(state1)
    print(temp)
    for i in range(16):
        state[i] = temp[2*i + 1]
    return state


def perm_nibble_16_inv(state):
    state1 = [0 for i in range(32)]
    for i in range(16):
        state1[2*i] = state[i]
    temp = [0 for i in range(32)]
    for i in range(32):
        temp[i] = state1[perm[i]]
    print(state1)
    print(temp)
    for i in range(16):
        state[i] = temp[2*i + 1]
    return state

if __name__ == '__main__':
    state = [2*i for i in range(16)]
    print(state)
    state = perm_nibble_16(state)
    print(state)
    state = [2*i for i in range(16)]
    print(state)
    state = perm_nibble_16_inv(state)
    print(state)

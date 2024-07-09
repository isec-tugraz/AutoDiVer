# The MIT License (MIT)
# 
# Copyright (c) 2016 Calvin McCoy
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

s_box = (0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2)

inv_s_box = (0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA)

p_layer_order = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38,
                 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13,
                 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]

block_size = 64

ROUND_LIMIT = 32


def round_function(state, key):
    new_state = state ^ key
    state_nibs = []
    for x in range(0, block_size, 4):
        nib = (new_state >> x) & 0xF
        sb_nib = s_box[nib]
        state_nibs.append(sb_nib)
    # print(state_nibs)

    state_bits = []
    for y in state_nibs:
        nib_bits = [1 if t == '1'else 0 for t in format(y, '04b')[::-1]]
        state_bits += nib_bits
    # print(state_bits)
    # print(len(state_bits))

    state_p_layer = [0 for _ in range(64)]
    for p_index, std_bits in enumerate(state_bits):
        state_p_layer[p_layer_order[p_index]] = std_bits

    # print(len(state_p_layer), state_p_layer)

    round_output = 0
    for index, ind_bit in enumerate(state_p_layer):
        round_output += (ind_bit << index)

    # print(format(round_output, '#016X'))

    # print('')
    return round_output


def key_function_80(key, round_count):
    key = ((key << 61) ^ (key >> 19)) & 0xFFFFFFFFFFFFFFFFFFFF
    upper_nibble = key >> 76
    upper_nibble = s_box[upper_nibble]
    key = (key & 0x0FFFFFFFFFFFFFFFFFFF) ^ (upper_nibble << 76) ^ (round_count << 15)

    return key


def key_function_128(key, round_count):
    key = ((key << 61) ^ (key >> 67)) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    upper_nibble_1 = key >> 124
    upper_nibble_2 = (key >> 120) & 0xF
    upper_nibble_1 = s_box[upper_nibble_1]
    upper_nibble_2 = s_box[upper_nibble_2]
    key = (key & 0x00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF) ^ (upper_nibble_1 << 124) ^ (upper_nibble_2 << 120) ^ (round_count << 62)
    return key


def present_enc80(plaintext: int, key: int, num_rounds: int = 31, *, do_final_key_xor=True):
    key_schedule = []
    round_state = plaintext
    current_round_key = key

    # Key schedule
    for rnd_cnt in range(num_rounds + 1):
        # print(format(round_key, '020X'))
        # print(format(round_key >> 16, '016X'))
        key_schedule.append(current_round_key >> 16)
        current_round_key = key_function_80(current_round_key, rnd_cnt + 1)

    for rnd in range(num_rounds):
        # print('Round:', rnd)
        # print('State:', format(round_state, '016X'))
        # print('R_Key:', format(key_schedule[rnd], '016X'))
        round_state = round_function(round_state, key_schedule[rnd])

    if do_final_key_xor:
        round_state ^= key_schedule[num_rounds]
    return round_state


def present_enc128(plaintext: int, key: int, num_rounds=31, *, do_final_key_xor=True):
    key_schedule = []
    round_state = plaintext
    current_round_key = key

    # Key schedule
    for rnd_cnt in range(num_rounds + 1):
        # print(format(round_key, '020X'))
        # print(format(round_key >> 16, '016X'))
        key_schedule.append(current_round_key >> 64)
        current_round_key = key_function_128(current_round_key, rnd_cnt + 1)

    for rnd in range(num_rounds):
        # print('Round:', rnd)
        # print('State:', format(round_state, '016X'))
        # print('R_Key:', format(key_schedule[rnd], '016X'))
        round_state = round_function(round_state, key_schedule[rnd])

    if do_final_key_xor:
        round_state ^= key_schedule[num_rounds]

    return round_state


if __name__ == '__main__':
    test_vectors_80 = [(0x00000000000000000000, 0x0000000000000000, 0x5579C1387B228445),
                       (0xFFFFFFFFFFFFFFFFFFFF, 0x0000000000000000, 0xE72C46C0F5945049),
                       (0x00000000000000000000, 0xFFFFFFFFFFFFFFFF, 0xA112FFC72F68417B),
                       (0xFFFFFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3333DCD3213210D2)]

    test_vectors_128 = [(0x00000000000000000000000000000000, 0x0000000000000000, 0x96db702a2e6900af),
                        (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0x0000000000000000, 0x13238c710272a5d8),
                        (0x00000000000000000000000000000000, 0xFFFFFFFFFFFFFFFF, 0x3c6019e5e5edd563),
                        (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x628d9fbd4218e5b4)]

    print('Testing 80-bit Key Vectors:')
    for test_case in test_vectors_80:
        ct = present_enc80(test_case[1], test_case[0])

        if ct == test_case[2]:
            print('Success', hex(ct))
        else:
            print('Failure', hex(ct))

    print('')
    print('Testing 128-bit Key Vectors:')
    for test_case in test_vectors_128:
        ct = present_enc128(test_case[1], test_case[0])

        if ct == test_case[2]:
            print('Success', hex(ct))
        else:
            print('Failure', hex(ct))

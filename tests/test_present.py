from random import randint
import pytest
from copy import copy

import numpy as np
from sat_toolkit.formula import XorCNF

from autodiver.cipher_model import count_solutions
from autodiver.present.present_cipher import present_enc80, present_enc128, key_function_80, key_function_128
from autodiver.present.present_model import Present80, PresentCharacteristic



test_vectors_80 = [(0x00000000000000000000, 0x0000000000000000, 0x5579C1387B228445),
                   (0xFFFFFFFFFFFFFFFFFFFF, 0x0000000000000000, 0xE72C46C0F5945049),
                   (0x00000000000000000000, 0xFFFFFFFFFFFFFFFF, 0xA112FFC72F68417B),
                   (0xFFFFFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3333DCD3213210D2)]

test_vectors_128 = [(0x00000000000000000000000000000000, 0x0000000000000000, 0x96db702a2e6900af),
                    (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0x0000000000000000, 0x13238c710272a5d8),
                    (0x00000000000000000000000000000000, 0xFFFFFFFFFFFFFFFF, 0x3c6019e5e5edd563),
                    (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x628d9fbd4218e5b4)]

@pytest.mark.parametrize("key,pt,ct_ref", test_vectors_80)
def test_present_enc80(key, pt, ct_ref):
    ct = present_enc80(pt, key)
    assert ct == ct_ref

@pytest.mark.parametrize("key,pt,ct_ref", test_vectors_128)
def test_present_enc128(key, pt, ct_ref):
    ct = present_enc128(pt, key)
    assert ct == ct_ref

def test_count_present80():
    num_rounds = 2
    sbi_delta = sbo_delta = np.zeros((num_rounds, 16), dtype=np.uint8)
    char = PresentCharacteristic(sbi_delta, sbo_delta)
    present = Present80(char)
    assert len(present.key) == 80

    cnf_new = copy(present.cnf)
    cnf_new += XorCNF.create_xor(present.key)
    cnf_new += XorCNF.create_xor(present.pt.flatten())
    sol_count = count_solutions(cnf_new, epsilon=0.2, delta=0.8)
    assert sol_count == 1

    cnf_all = copy(present.cnf)
    sol_count = count_solutions(cnf_all, epsilon=0.2, delta=0.8)
    assert sol_count == 2**80 * 2**64
    

def test_zero_characteristic_present80():
    num_rounds = 7
    sbi_delta = sbo_delta = np.zeros((num_rounds, 16), dtype=np.uint8)
    char = PresentCharacteristic(sbi_delta, sbo_delta)
    present = Present80(char)

    # present.cnf += XorCNF.create_xor(present.key)
    # present.cnf += XorCNF.create_xor(present.pt.flatten())

    print(present.key.shape, present.key.dtype)
    model = present.solve()
    
    key = model.key # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    pt = model.pt # type: ignore

    round_keys = model.round_keys # type: ignore
    long_round_keys = model.long_round_keys # type: ignore

    print('key', key.shape, key.dtype)
    print('sbi', sbi.shape, sbi.dtype)
    print('sbo', sbo.shape, sbo.dtype)

    pt = sum(int(x) << i * 4 for i, x in enumerate(pt))
    ct = sum(int(x) << i * 4 for i, x in enumerate(sbi[-1]))
    key = sum(int(x) << i * 8 for i, x in enumerate(key))

    ct_ref = present_enc80(pt, key, num_rounds=num_rounds)
    assert ct == ct_ref

    print(f'pt: {pt:016x}')
    print(f'key: {key:020x}')
    print(f'ct: {ct:016x}')

    print(' round_keys '.center(80, '-'))
    prev_rk = sum(int(x) << i * 8 for i, x in enumerate(long_round_keys[0]))
    ref_rk = key
    print(f'round {0}: {prev_rk:020x}, {ref_rk:020x}')
    assert prev_rk == ref_rk
    for rnd, rk in enumerate(long_round_keys[1:], start=1):
        rk = sum(int(x) << i * 8 for i, x in enumerate(rk))
        ref_rk = key_function_80(prev_rk, rnd)
        print(f'round {rnd}: {rk:016x}, {ref_rk:016x}')
        assert rk == ref_rk
        prev_rk = rk

    
    for rk, long_rk in zip(round_keys, long_round_keys):
        long_rk = sum(int(x) << i * 8 for i, x in enumerate(long_rk))
        assert np.all(rk == long_rk >> 16)

    
    print(' internal state '.center(80, '-'))
    for i in range(num_rounds):
        this_sbi = sum(int(x) << i * 4 for i, x in enumerate(sbi[i]))
        this_sbo = sum(int(x) << i * 4 for i, x in enumerate(sbo[i]))
        print(f'round {i}: {this_sbi:016x} -> {this_sbo:016x}')

    print(' comparison with ref '.center(80, '-'))
    for i in range(num_rounds + 1):
        output = sum(int(x) << i * 4 for i, x in enumerate(sbi[i]))
        ref_output = present_enc80(pt, key, num_rounds=i, do_final_key_xor=True)
        print(f'{i} rounds: {output:016x}, {ref_output:016x}')

    ct_ref = present_enc80(pt, key, num_rounds=num_rounds)
    assert ct == ct_ref
    print(f'{ct:016X} == {ct_ref:016X}')

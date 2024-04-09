from random import randint
import numpy as np
import pytest
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.warp128.warp128_model import WARP128
from differential_verification.warp128.warp_cipher import warp_enc
from differential_verification.warp128.util import get_round_in_out, perm_nibble_inv
from sat_toolkit.formula import CNF
from icecream import ic
def print_state(S, state = "s"):
    print(state, ":", end = " ")
    for s in S:
        print(hex(s)[2:], end = "")
    print("")
def str_to_state(A_str):
    state = []
    for a in A_str:
        b = int(a, 16)
        state.append(b)
    return np.asarray(state, np.uint8)
def Add(A, B):
    assert A.shape == B.shape
    state = []
    for i in range(len(A)):
        s = A[i] ^ B[i]
        state.append(s)
    return np.asarray(state, np.uint8)
testvectors = [
(np.asarray([0xa, 0xf, 0x6, 0xc, 0xd, 0xd, 0x9, 0x0, 0xf, 0xc, 0x5, 0xa, 0x6, 0xe, 0xa, 0xa, 0x8, 0x9, 0x7, 0xb, 0xc, 0xd, 0x1, 0x2, 0x0, 0x8, 0xd, 0x3, 0x9, 0x1, 0xe, 0x1], dtype=np.uint8),
np.asarray([0x0, 0xa, 0xc, 0xd, 0x0, 0x2, 0x2, 0xf, 0x6, 0x8, 0x0, 0xa, 0x5, 0x4, 0x7, 0xf, 0xe, 0xe, 0x0, 0x3, 0xc, 0x0, 0x8, 0x6, 0x7, 0xb, 0x0, 0x9, 0xe, 0x3, 0xd, 0x7], dtype=np.uint8),
np.asarray([0x6, 0x1, 0x2, 0x3, 0x9, 0x9, 0x5, 0xf, 0x1, 0x9, 0x2, 0x4, 0xd, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x4, 0x1, 0xa, 0xc, 0xd, 0xd, 0x0, 0x5, 0x8, 0xd, 0xd, 0x4, 0x6], dtype=np.uint8) )
]
@pytest.mark.parametrize("pt,key,ct_ref", testvectors)
def test_tv(pt, key, ct_ref):
    print_state(pt, "M")
    print_state(key, "K")
    print_state(ct_ref, "C")
    ct = warp_enc(pt, key, 41)
    print_state(ct, "C")
    assert np.all(ct == ct_ref)
def test_zero_characteristic():
    numrounds = 3
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
    char.sbox_in = sbi
    char.sbox_out = sbo
    char.num_rounds = numrounds
    warp = WARP128(char)
    # print(warp.cnf)
    num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 128)
    for bit_var in warp.key.flatten():
        warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << 128
    for bit_var in warp.sbox_in[0].flatten():
        warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    for bit_var in warp.X.flatten():
        warp.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    model = warp.solve()
    key = model.key # type: ignore
    print_state(key)
    pt = model.pt # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    assert np.all(warp.sbox[sbi[:warp.num_rounds]] == sbo)
    rout = get_round_in_out(numrounds, model)
    # print_state(rin[0])
    # assert np.all(rin[0] == pt)
    print_state(pt)
    for r, round_out in enumerate(rout):
        ref = warp_enc(pt, key, r+1)
        print("----------------------------------")
        # print_state(sbi[r])
        # if r != warp.num_rounds - 1:
        #     print_state(perm_nibble_16_inv(sbi[r+1]))
        # else:
        #     print_state(model.Y)
        print_state(ref)
        print_state(round_out)
        print("----------------------------------")
        assert np.all(round_out == ref)
    num_solutions = count_solutions(warp.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1
def test_nonzero_characteristic():
    char =  (('0000000000041000', '0000000000022000'),
             ('0020000000000000', '0040000000000000'),
             ('1000000000000000', '2000000000000000'),
             ('0000000000000000', '0000000000000000'),
             ('0000000000000100', '0000000000000200'),
             ('0000000020000000', '0000000040000000'),
             ('0000000000040001', '0000000000020002'),
             ('0000020000002200', '00000c000000c400'))
    inds =  ('00000000000002000000004212000000', '00002400000100000000000000000000',
             '12000000000000000000000000000000', '00000000000000000000000000000001',
             '00000000000000000000000000100000', '00000000000000002001000000000000',
             '00000000000000020000004000000010', '00000401002000000000000020200000',
             'c00200001002400040c2000000000000')
    sbi_delta = np.array([[int(x, 16) for x in in_out[0]] for in_out in char], dtype=np.uint8)
    sbo_delta = np.array([[int(x, 16) for x in in_out[1]] for in_out in char], dtype=np.uint8)
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
    warp = WARP128(char)
    model = warp.solve()
    key = model.key # type: ignore
    print_state(key)
    pt = model.pt # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    rout = get_round_in_out(warp.num_rounds, model)
    assert np.all(warp.sbox[sbi[:warp.num_rounds]] == sbo)
    print_state(pt, "pt")
    delta0 = str_to_state(inds[0])
    for r in range(1, char.num_rounds):
        print(f' round {r} '.center(80, '='))
        ref = warp_enc(pt, key, r)
        ref_xor = warp_enc(Add(pt, delta0), key, r)
        assert np.all(rout[r-1] == ref)
        found_diff = Add(ref, ref_xor)
        print_state(found_diff)
        expected_diff = perm_nibble_inv(str_to_state(inds[r]))
        print_state(expected_diff)
        assert np.all(expected_diff == found_diff)
if __name__ == "__main__":
    test_zero_characteristic()
    test_nonzero_characteristic()
    # for tv in midori64_testvectors:
    #     test_tv(*tv)
from random import seed, randint
from differential_verification.cipher_model import DifferentialCharacteristic, count_solutions
from differential_verification.speedy192.speedy192_model import Speedy192
from differential_verification.speedy192.speedy_cipher import speedy192_enc
from differential_verification.speedy192.util import Add, prepare_round_keys
import numpy as np
import pytest
from sat_toolkit.formula import CNF
from icecream import ic
#0th bit is the LSB
def print_state(key, s = "state"):
    print(s, end = ": " )
    for k in key:
        # print(k)
        assert k <= 64
        print(hex(k)[2:].zfill(2), end = " ")
    print("")
# speedy192_testvectors = [
#     ("51084ce6e73a5ca2ec87d7babc297543", "687ded3b3c85b3f35b1009863e2a8cbf", "1e0ac4fddff71b4c1801b73ee4afc83d"),
# ]
# speedy192_testvectors = [
#     (bytes.fromhex(pt), bytes.fromhex(key), bytes.fromhex(ct_ref)) for pt, key, ct_ref in speedy192_testvectors
# ]
# @pytest.mark.parametrize("pt,key,ct_ref", speedy192_testvectors)
# def test_tv(pt: bytes, key: bytes, ct_ref: bytes):
#     print(f'{pt=}')
#     print(f'{key=}')
#     print(f'{ct_ref=}')
#     ct = speedy192_enc(pt, key, 7)
#     assert ct == ct_ref
def test_zero_characteristic():
    seed("test_speedy192::test_zero_characteristic")
    numrounds = 2
    sbi_delta = sbo_delta = np.zeros((2*numrounds, 32), dtype=np.uint8)
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
    speedy = Speedy192(char)
    # num_solutions = count_solutions(speedy.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1 << (192 + 192)
    for bit_var in speedy.key.flatten():
        speedy.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
        # speedy.cnf += CNF([-bit_var, 0])
    # num_solutions = count_solutions(speedy.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1 << 192
    for bit_var in speedy.sbox_out[0, 1:].flatten():
        speedy.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
        # speedy.cnf += CNF([-bit_var, 0])
    for bit_var in speedy.sbox_out[0, :1].flatten():
        speedy.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
        # speedy.cnf += CNF([bit_var, 0])
    model = speedy.solve()
    key = model.key[0] # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    mco = model.mc_out # type: ignore
    assert np.all(speedy.sbox[sbi[:speedy.num_rounds]] == sbo)
    print_state(key)
    pt = Add(sbi[0], key)
    print_state(pt)
    round_keys = prepare_round_keys(key)
    for i in range(speedy.num_rounds_full):
        print_state(round_keys[i], "rkeys{}".format(i))
    #Note that inside the model ther is not key addition at the start and end
    ref = speedy192_enc(pt, key, numrounds)
    ref = Add(ref, round_keys[numrounds])
    out = sbo[2*(numrounds-1) + 1]
    print_state(ref, "ref")
    print_state(out, "sbo")
    assert np.all(ref == out)
    num_solutions = count_solutions(speedy.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1
def test_nonzero_characteristic():
    char =( (( 1,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0),
             (16,  0,  0,  0, 16, 16,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0, 16,  0,  0,  0)),
            (( 0,  0,  0, 16, 16,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0, 16,  0,  0,  0, 16),
             ( 0,  0,  0,  4,  4,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  4,  0,  0,  0,  4)),
            (( 0,  4,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4),
             ( 0, 16,  0, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0, 17,  0, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16)),
            ((16,  0, 16,  0,  0,  0,  0,  0,  1,  0,  0,  2, 16,  0,  0, 32,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0),
             ( 4,  0,  4,  0,  0,  0,  0,  0,  4,  0,  0,  4,  4,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0)) )
    # char =(
    #         (( 0,  4,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4),
    #          ( 0, 16,  0, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0, 17,  0, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16)),
    #         ((16,  0, 16,  0,  0,  0,  0,  0,  1,  0,  0,  2, 16,  0,  0, 32,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0),
    #          ( 4,  0,  4,  0,  0,  0,  0,  0,  4,  0,  0,  4,  4,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0)) )
    numrounds = 2  #Number of full rounds
    sbi_delta = np.array([[x for x in in_out[0]] for in_out in char], dtype=np.uint8)
    sbo_delta = np.array([[x for x in in_out[1]] for in_out in char], dtype=np.uint8)
    char = DifferentialCharacteristic(sbi_delta, sbo_delta)
    speedy = Speedy192(char)
    with pytest.raises(ValueError):
        model = speedy.solve()
        key = model.key[0] # type: ignore
        sbi = model.sbox_in # type: ignore
        sbo = model.sbox_out # type: ignore
        mco = model.mc_out # type: ignore
        assert np.all(speedy.sbox[sbi[:speedy.num_rounds]] == sbo)
        print_state(key)
        pt = Add(sbi[0], key)
        print_state(pt)
        ref = speedy192_enc(pt, key, numrounds)
        for r in range(numrounds):
            print_state(sbi[2*r], "sbi")
            print_state(sbo[2*r], "sbo")
            print_state(sbi[2*r+1],"sbi")
            print_state(sbo[2*r+1],"sbo")
            if r != (numrounds - 1):
                print_state(mco[r],    "mco")
        out = sbo[2*(numrounds-1) + 1]
        print_state(ref, "ref")
        print_state(out, "sbo")
        assert np.all(ref == out)
        num_solutions = count_solutions(speedy.cnf, epsilon=0.8, delta=0.2, verbosity=0)
        assert num_solutions == 1
if __name__ == "__main__":
    test_zero_characteristic()
    test_nonzero_characteristic()
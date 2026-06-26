from random import seed, randint

from autodiver.cipher_model import count_solutions, UnsatException
from autodiver.speedy192.speedy192_model import Speedy192, Speedy192Characteristic
from autodiver_ciphers.speedy192.speedy_cipher import speedy192_enc


from autodiver.speedy192.util import Add, prepare_round_keys

import numpy as np
import pytest
from sat_toolkit.formula import CNF
from pathlib import Path

from icecream import ic


#0th bit is the LSB
def print_state(key, s = "state"):
    print(s, end = ": " )
    for k in key:
        # print(k)
        assert k <= 64
        print(hex(k)[2:].zfill(2), end = " ")
    print("")

def str_state(S):
    state = [int(S[i:i+2],16) for i in range(0, len(S), 2)]
    print(state)
    return np.asarray(state, dtype=np.uint8)

speedy192_testvectors = [
    ("281329230905040703240e0228273c2629001a023c3f3a1f3d28002834243f1b", "1d24310f182513212f3f08083a15212210283e2e340116043d02013a1f281137", "3b13340e281131021e3d0c251c0d3d01312f35263a3b3d09052e1d203b102503"),
]
speedy192_testvectors = [
    (str_state(pt), str_state(key), str_state(ct_ref)) for pt, key, ct_ref in speedy192_testvectors
]

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
TRAILS_DIR: Path = REPO_ROOT / "trails" / "speedy192"


@pytest.mark.parametrize("pt,key,ct_ref", speedy192_testvectors)
def test_tv(pt, key, ct_ref):
    ct = speedy192_enc(pt, key, 7)
    assert np.all(ct == ct_ref)

def test_zero_characteristic():
    seed("test_speedy192::test_zero_characteristic")
    numrounds = 2
    sbi_delta = sbo_delta = np.zeros((2*numrounds, 32), dtype=np.uint8)
    char = Speedy192Characteristic(sbi_delta, sbo_delta)
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

    model = speedy.solve(seed=9732)

    key = model.key[0] # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    mco = model.mc_out # type: ignore

    assert np.all(speedy.sbox[sbi[:speedy.num_rounds]] == sbo)
    print_state(key)
    pt = sbi[0] ^ key
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

def test_nonzero_characteristic_unsat():
    char = Speedy192Characteristic.load(TRAILS_DIR / "speedy192_BDBN23_r5.txt")
    speedy = Speedy192(char)

    print(f'ddt probability: 2^{char.log2_ddt_probability():.1f}')

    with pytest.raises(UnsatException):
        model = speedy.solve(seed=1)

def test_nonzero_characteristic_sat():
    char = Speedy192Characteristic.load(TRAILS_DIR / "speedy192_demo.txt")
    numrounds = char.num_rounds // 2 # number of full rounds

    speedy = Speedy192(char)

    print(f'ddt probability: 2^{char.log2_ddt_probability():.1f}')

    model = speedy.solve(seed=1)
    key = model.key[0] # type: ignore
    sbi = model.sbox_in # type: ignore
    sbo = model.sbox_out # type: ignore
    mco = model.mc_out # type: ignore

    assert np.all(speedy.sbox[sbi[:speedy.num_rounds]] == sbo)
    print_state(key, "key")
    pt = sbi[0] ^ key
    print_state(pt, "pt")
    round_keys = prepare_round_keys(key)
    for i in range(speedy.num_rounds_full):
        print_state(round_keys[i], "rkeys{}".format(i))

    #Note that inside the model there is not key addition at the start and end
    ref = speedy192_enc(pt, key, numrounds)
    ref = ref ^ round_keys[numrounds]
    out = sbo[2*(numrounds-1) + 1]
    print_state(ref, "ref")
    print_state(out, "sbo")
    assert np.all(ref == out)

    ref_xor = speedy192_enc(pt ^ char.sbox_in[0], key, numrounds)
    ref_xor = ref_xor ^ round_keys[numrounds]
    found_diff = ref ^ ref_xor
    print_state(found_diff)
    expected_diff = char.sbox_out[2*(numrounds-1) + 1]
    print_state(expected_diff)
    assert np.all(expected_diff == found_diff)

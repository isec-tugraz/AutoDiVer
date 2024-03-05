from random import randint
import numpy as np
from cipher_model import DifferentialCharacteristic, count_solutions
from midori.midori import Midori64
from sat_toolkit.formula import CNF
# from pyximport import install
# install()
# from gift64.gift_cipher import gift64_enc
def test_zero_characteristic():
    numrounds = 5
    sbi = sbo = np.zeros((numrounds, 16), dtype=np.uint8)
    char = DifferentialCharacteristic.__new__(DifferentialCharacteristic)
    char.sbox_in = sbi
    char.sbox_out = sbo
    char.num_rounds = numrounds
    midori = Midori64(char)
    num_solutions = count_solutions(midori.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << (128 + 64)
    for bit_var in midori.key.flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    assert num_solutions == 1 << 64
    for bit_var in midori.sbox_in[0].flatten():
        midori.cnf += CNF([bit_var * (-1)**randint(0,1), 0])
    model = midori.solve()
    # key = model.key # type: ignore
    # sbi = model.sbox_in # type: ignore
    # sbo = model.sbox_out # type: ignore
    # assert np.all(gift.sbox[sbi[:gift.num_rounds]] == sbo)
    # for r, round_sbi in enumerate(sbi):
    #     ref = gift64_enc(sbi[0], key, r)
    #     assert np.all(round_sbi == ref)
    # num_solutions = count_solutions(gift.cnf, epsilon=0.8, delta=0.2, verbosity=0)
    # assert num_solutions == 1
if __name__ == "__main__":
    test_zero_characteristic()
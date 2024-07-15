from copy import copy
import random
import pytest
import numpy as np
from sat_toolkit.formula import CNF, XorCNF
from autodiver.skinny.skinny_model import Skinny128, Skinny128Characteristic
from autodiver.skinny.constants import do_mix_cols, do_inv_mix_cols, do_shift_rows, expanded_rc, update_tweakey, tweakey_mask


def test_zero_characteristic():
    numrounds = 4
    sbox_in_delta = sbox_out_delta   = np.zeros((numrounds, 4, 4), dtype=np.uint8)
    tweakeys = np.zeros((numrounds, 3, 4, 4), dtype=np.uint8)
    char = Skinny128Characteristic(sbox_in_delta, sbox_out_delta, tweakeys)
    cipher = Skinny128(char)
    model = cipher.solve(seed=9789)
    sbox_in = model.sbox_in #type: ignore
    sbox_out = model.sbox_out #type: ignore
    round_tweakeys = model.round_tweakeys #type: ignore
    key, tk2, tk3 = round_tweakeys[0]
    sbox_in = sbox_in[:-1]
    sbox = cipher.sbox
    assert np.all(sbox[sbox_in] == sbox_out)
    # sanity check result
    rtk = np.array([key, tk2, tk3])
    for i in range(numrounds):
        sbi = sbox_in[i]
        sbo = sbox_out[i]
        assert np.all(sbo == sbox[sbi])
        if i + 1 in range(numrounds):
            rc = expanded_rc[i]
            sbo = sbox_out[i]
            mc_output = sbox_in[i + 1]
            this_rtk = np.bitwise_xor.reduce(rtk, axis=0) & tweakey_mask
            mc_input = do_shift_rows(sbo ^ this_rtk ^ rc)
            assert np.all(mc_output == do_mix_cols(mc_input))
        rtk = update_tweakey(rtk)


def test_unique_solution():
    random.seed("test_skinny128::test_unique_solution")
    numrounds = 4
    sbox_in_delta = sbox_out_delta   = np.zeros((numrounds, 4, 4), dtype=np.uint8)
    tweakeys = np.zeros((numrounds, 3, 4, 4), dtype=np.uint8)
    char = Skinny128Characteristic(sbox_in_delta, sbox_out_delta, tweakeys)
    cipher = Skinny128(char)
    inputs = cipher.pt.flatten().tolist() + cipher.key.flatten().tolist() + cipher.tweak.flatten().tolist()
    ct_idxes = cipher.sbox_out[-1].flatten().tolist()
    assert len(inputs) == 128 + 128 + 256
    for _ in range(10):
        cnf = copy(cipher.cnf)
        extra_clauses = np.zeros(len(inputs) * 2, np.int32)
        for i, inp in enumerate(inputs):
            extra_clauses[i * 2] = inp * (-1)**random.randint(0, 1)
            extra_clauses[i * 2 + 1] = 0
        cnf += CNF(extra_clauses)
        is_sat, model = cnf.solve_dimacs()
        assert(is_sat)
        ct = model[ct_idxes]
        extra_clause = np.zeros(len(ct) + 1, np.int32)
        extra_clause[:-1] = ct_idxes * (-1)**(ct)
        cnf += CNF(extra_clause)
        is_sat, model = cnf.solve_dimacs()
        assert(not is_sat)


@pytest.mark.parametrize("rounds", [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 10)])
def test_nonzero_characteristic(rounds):
    numrounds = len(range(rounds.start, rounds.stop))
    sbox_in = np.array(bytearray.fromhex(
        "00000000000000000000000000000000"
        "0000bf000000bf00420000b70000bf00"
        "0000004b0000cb4b00cbcb0e00cb004b"
        "000048000037003600000e00d2000000"
        "aa00000c0000000000000000aa000000"
        "00000034000000000000004b00000000"
        "0000000000fd00000000000000000000"
        "00000000000000000000e60000000000"
        "00000000cd0000000000004900000000"
        "0000000000ac00000000000000000000"
    )).reshape(10, 4, 4)[rounds]
    sbox_out = np.array(bytearray.fromhex(
        "00000000000000000000000000000000"
        "0000cb0000004500cb0000cb0000cb00"
        "00000036000036d20036d23700d20048"
        "0000cd00004900e30000aa000c000000"
        "b60000ac000000000000000034000000"
        "000000fd00000000000000fd00000000"
        "00000000006200000000000000000000"
        "00000000000000000000cd0000000000"
        "000000001a000000000000ac00000000"
        "00000000004b00000000000000000000"
    )).reshape(10, 4, 4)[rounds]
    tweakeys = np.array(bytearray.fromhex(
        "00000000000000000000000000000000" "00006f000000dd090096000049490000" "0000d00000006a4b005b0000b3b30000"
        "00000000000000000000000000000000" "2d0000920000920000006f000000dd09" "2d0000d90000d9000000d00000006a4b"
        "00000000000000000000000000000000" "00120000dfbb00002d00009200009200" "00250000e8b500002d0000d90000d900"
        "00000000000000000000000000000000" "00005b000025002500120000dfbb0000" "00009600006c006c00250000e8b50000"
        "00000000000000000000000000000000" "240000760000bf0000005b0000250025" "920000da0000f40000009600006c006c"
        "00000000000000000000000000000000" "004b004bb6000000240000760000bf00" "00b600b64b000000920000da0000f400"
        "00000000000000000000000000000000" "00004900007e00ed004b004bb6000000" "0000490000fa00ed00b600b64b000000"
        "00000000000000000000000000000000" "9600000000006c9600004900007e00ed" "5b0000000000255b0000490000fa00ed"
        "00000000000000000000000000000000" "00da00fd920000009600000000006c96" "007600fd240000005b0000000000255b"
        "00000000000000000000000000000000" "002d2d0000d9000000da00fd92000000" "002d2d0000920000007600fd24000000"
    )).reshape(10, 3, 4, 4)[rounds]

    char = Skinny128Characteristic(sbox_in, sbox_out, tweakeys)
    cipher = Skinny128(char)
    assert char.num_rounds == numrounds
    model = cipher.solve(seed=2006)
    sbox_in = model.sbox_in #type: ignore
    sbox_out = model.sbox_out #type: ignore
    round_tweakeys = model.round_tweakeys #type: ignore
    print(round_tweakeys.shape)
    key, tk2, tk3 = round_tweakeys[0]
    sbox = cipher.sbox
    assert np.all(sbox[sbox_in[:-1]] == sbox_out)
    np.set_printoptions(formatter={'int': lambda x: f'{x:02x}'})
    # sanity check result
    rtk = np.array([key, tk2, tk3])
    for i in range(numrounds):
        sbi = sbox_in[i]
        sbo = sbox_out[i]
        assert np.all(sbo == sbox[sbi])
        actual_rtk = round_tweakeys[i]
        print(f"round {i}".center(80, '-'))
        print(rtk.shape, actual_rtk.shape)
        print(rtk ^ actual_rtk)
        assert np.all(rtk == actual_rtk)
        rc = expanded_rc[i]
        sbo = sbox_out[i]
        mc_output = sbox_in[i + 1]
        this_rtk = np.bitwise_xor.reduce(rtk, axis=0) & tweakey_mask
        mc_input = do_shift_rows(sbo ^ this_rtk ^ rc)
        print(f"input\n{mc_input}")
        print(f"output ref\n{do_mix_cols(mc_input)}")
        print(f"output act\n{mc_output}")
        print(f"output diff\n{do_mix_cols(mc_input) ^ mc_output}")
        assert np.all(mc_output == do_mix_cols(mc_input))
        rtk = update_tweakey(rtk)



@pytest.mark.parametrize("rounds", [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(15, 17)])
def test_acns2021_characteristic(rounds):
    numrounds = len(range(rounds.start, rounds.stop))
    sbox_in = np.array(bytearray.fromhex(
        "00000200002000000800000000000808"
        "00100000000008000000000000001000"
        "00000000004000000000001000400000"
        "04400005000000050040040000400005"
        "00040505050400010000000404040005"
        "01000000000101010300010000000101"
        "00B30000200000000000002000002000"
        "00000000000000000000000000800000"
        "03000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000002900000000"
        "00300000000000000030000000300000"
    )).reshape(-1, 4, 4)[rounds]
    sbox_out = np.array(bytearray.fromhex(
        "00000800009200001800000000001010"
        "00400000000010000000000000004000"
        "00000000000400000000004000040000"
        "05040001000000010004010000040005"
        "00010101010100280000000101010001"
        "2000000000202020200020000000B320"
        "00EE0000800000000000008000008000"
        "00000000000000000000000000030000"
        "29000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000000000000000"
        "00000000000000000000003000000000"
        "00400000000000000040000000400000"
    )).reshape(-1, 4, 4)[rounds]
    tweakeys = np.array(bytearray.fromhex(
        "0000000000BA00000000000000000000" "00000000004300000000000000000000" "00000000007300000000000000000000"
        "00000000000000000000000000BA0000" "00000000000000000000000000430000" "00000000000000000000000000730000"
        "000000BA000000000000000000000000" "00000086000000000000000000000000" "00000039000000000000000000000000"
        "0000000000000000000000BA00000000" "00000000000000000000008600000000" "00000000000000000000003900000000"
        "00000000000000BA0000000000000000" "000000000000000D0000000000000000" "000000000000009C0000000000000000"
        "000000000000000000000000000000BA" "0000000000000000000000000000000D" "0000000000000000000000000000009C"
        "00BA0000000000000000000000000000" "001A0000000000000000000000000000" "004E0000000000000000000000000000"
        "000000000000000000BA000000000000" "0000000000000000001A000000000000" "0000000000000000004E000000000000"
        "BA000000000000000000000000000000" "34000000000000000000000000000000" "A7000000000000000000000000000000"
        "0000000000000000BA00000000000000" "00000000000000003400000000000000" "0000000000000000A700000000000000"
        "0000BA00000000000000000000000000" "00006900000000000000000000000000" "0000D300000000000000000000000000"
        "00000000000000000000BA0000000000" "00000000000000000000690000000000" "00000000000000000000D30000000000"
        "00000000BA0000000000000000000000" "00000000D30000000000000000000000" "00000000690000000000000000000000"
        "000000000000000000000000BA000000" "000000000000000000000000D3000000" "00000000000000000000000069000000"
        "000000000000BA000000000000000000" "000000000000A7000000000000000000" "00000000000034000000000000000000"
        "0000000000000000000000000000BA00" "0000000000000000000000000000A700" "00000000000000000000000000003400"
        "0000000000ba00000000000000000000" "00000000004e00000000000000000000" "00000000001a00000000000000000000"
    )).reshape(-1, 3, 4, 4)[rounds]

    char = Skinny128Characteristic(sbox_in, sbox_out, tweakeys)
    cipher = Skinny128(char)

    assert char.num_rounds == numrounds
    model = cipher.solve(seed=1159)
    sbox_in = model.sbox_in #type: ignore
    sbox_out = model.sbox_out #type: ignore
    round_tweakeys = model.round_tweakeys #type: ignore
    print(round_tweakeys.shape)
    key, tk2, tk3 = round_tweakeys[0]
    sbox = cipher.sbox
    assert np.all(sbox[sbox_in[:-1]] == sbox_out)
    np.set_printoptions(formatter={'int': lambda x: f'{x:02x}'})
    # sanity check result
    rtk = np.array([key, tk2, tk3])
    for i in range(numrounds):
        sbi = sbox_in[i]
        sbo = sbox_out[i]
        assert np.all(sbo == sbox[sbi])
        actual_rtk = round_tweakeys[i]
        print(f"round {i}".center(80, '-'))
        print(rtk.shape, actual_rtk.shape)
        print(rtk ^ actual_rtk)
        assert np.all(rtk == actual_rtk)
        rc = expanded_rc[i]
        sbo = sbox_out[i]
        mc_output = sbox_in[i + 1]
        this_rtk = np.bitwise_xor.reduce(rtk, axis=0) & tweakey_mask
        mc_input = do_shift_rows(sbo ^ this_rtk ^ rc)
        print(f"input\n{mc_input}")
        print(f"output ref\n{do_mix_cols(mc_input)}")
        print(f"output act\n{mc_output}")
        print(f"output diff\n{do_mix_cols(mc_input) ^ mc_output}")
        assert np.all(mc_output == do_mix_cols(mc_input))
        rtk = update_tweakey(rtk)

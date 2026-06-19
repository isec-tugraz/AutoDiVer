import subprocess as sp
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fix_chdir(monkeypatch):
    file_path = Path(__file__).resolve()
    project_root = file_path.parent.parent
    assert (project_root / 'trails').is_dir()
    monkeypatch.chdir(project_root)


def test_count_tweakey_space_lin():
    expected_output = """
INFO gathered keys span affine space of dimension 111
INFO solving for counterexample
INFO RESULT no counterexample found -> conditions on key are necessary
INFO RESULT key[0, 0, 0, 3] ⊕ key[0, 1, 0, 3] = 1
INFO RESULT key[0, 0, 3, 3] ⊕ key[0, 1, 3, 3] ⊕ key[0, 3, 3, 3] = 0
INFO RESULT key[0, 1, 1, 3] ⊕ key[0, 3, 1, 3] = 0
INFO RESULT key[0, 1, 2, 3] ⊕ key[0, 3, 2, 3] = 1
INFO RESULT key[0, 3, 0, 3] = 0
INFO RESULT key[1, 0, 1, 2] ⊕ key[1, 1, 1, 2] ⊕ key[1, 3, 1, 2] = 0
INFO RESULT key[1, 0, 1, 3] ⊕ key[1, 1, 1, 3] ⊕ key[1, 3, 1, 3] = 0
INFO RESULT key[1, 0, 3, 2] ⊕ key[1, 1, 3, 2] = 0
INFO RESULT key[1, 0, 3, 3] ⊕ key[1, 1, 3, 3] = 0
INFO RESULT key[1, 1, 0, 0] = 1
INFO RESULT key[1, 1, 0, 3] = 0
INFO RESULT key[1, 2, 0, 0] ⊕ key[1, 3, 0, 0] = 0
INFO RESULT key[1, 2, 0, 3] ⊕ key[1, 3, 0, 3] = 0
INFO RESULT key[1, 2, 1, 2] ⊕ key[1, 3, 1, 2] = 0
INFO RESULT key[1, 2, 1, 3] ⊕ key[1, 3, 1, 3] = 0
INFO RESULT key[1, 3, 2, 2] = 0
INFO RESULT key[1, 3, 2, 3] = 0
""".lstrip()

    output = sp.check_output('autodiver midori64 trails/midori64/midori64_zhww_r5_1.npz count-tweakeys-lin', shell=True, stderr=sp.STDOUT, text=True)
    assert expected_output in output


# One (cipher, characteristic) pair per supported cipher. The base ciphers use a
# short characteristic; the long-key/full-key variants use a longer one since
# they are quick to solve regardless. rectangle128 is excluded because solving it
# is slow; rectangle-long-key is covered instead.
SOLVE_CASES = [
    ('ascon', 'trails/ascon/ascon_dems15_table_3.npz'),
    ('gift64', 'trails/gift64/gift64_toy_char.txt'),
    ('gift64-full-key', 'trails/gift64/gift64_lwzz19_r9_table_2.txt'),
    ('gift128', 'trails/gift128/gift128_r12_LLL+_table5/gift128_r01_LLL+_table_5.txt'),
    ('gift128-full-key', 'trails/gift128/gift128_r12_LLL+_table_5.txt'),
    ('midori64', 'trails/midori64/midori64_zhww_r5_1.npz'),
    ('midori64-long-key', 'trails/midori64/midori64_zhww_r5_1.npz'),
    ('midori128', 'trails/midori128/midori128_TAY16_table_3.npz'),
    ('midori128-long-key', 'trails/midori128/midori128_CXTQ23_table_4.npz'),
    ('present80', 'trails/present/present_Wang08_table_7_r14/present_Wang08_table_7_r01.npz'),
    ('present-long-key', 'trails/present/present_Wang08_table_7_r14.npz'),
    ('pyjamask', 'trails/pyjamask96/pyjamask_char1_r1.npz'),
    ('pyjamask-long-key', 'trails/pyjamask96/pyjamask_char1.npz'),
    ('rectangle-long-key', 'trails/rectangle/rectangle_ref_table_1.txt'),
    ('skinny64', 'trails/skinny64/skinny64_tk3_ddh+21_r15_table_9.npz'),
    ('skinny64-long-key', 'trails/skinny64/skinny64_tk3_ddh+21_r15_table_9.npz'),
    ('skinny128', 'trails/skinny128/skinny128_tk3_NPE23_r10_figure_8.npz'),
    ('skinny128-long-key', 'trails/skinny128/skinny128_tk3_ddh+21_r17_table_12_fixed.npz'),
    ('speck32-long-key', 'trails/speck/speck32_r9_ALLW14_table_7.npz'),
    ('speck48-long-key', 'trails/speck/speck48_r11_BR22_table_22_1a.npz'),
    ('speck64-long-key', 'trails/speck/speck64_r15_BR22_table_22b.npz'),
    ('speck96-long-key', 'trails/speck/speck96_r15_BR22_table_22c.npz'),
    ('speck128-long-key', 'trails/speck/speck128_r20_BR22_table_23a.npz'),
    ('speedy192', 'trails/speedy192/speedy192_demo.txt'),
    ('warp', 'trails/warp/warp_Hadipour_22.txt'),
]


@pytest.mark.parametrize('cipher,characteristic', SOLVE_CASES, ids=[c[0] for c in SOLVE_CASES])
def test_solve(cipher, characteristic):
    "test that autodiver CIPHER CHAR solve exists without errors"
    sp.run(['autodiver', cipher, characteristic, 'solve'], check=True, stderr=sp.STDOUT, stdout=sp.PIPE, text=True)

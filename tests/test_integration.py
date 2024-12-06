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

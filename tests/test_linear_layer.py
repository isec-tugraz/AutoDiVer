from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from autodiver.characteristic import DifferentialCharacteristic
from autodiver.ascon.ascon_model import AsconCharacteristic
from autodiver.gift.gift_characteristic import Gift64Characteristic, Gift128Characteristic
from autodiver.midori64.midori64_model import Midori64Characteristic
from autodiver.midori128.midori128_model import Midori128Characteristic
from autodiver.present.present_characteristic import PresentCharacteristic
from autodiver.pyjamask.pyjamask96_model import Pyjamask96Characteristic
from autodiver.rectangle128.rectangle_model import RectangleCharacteristic
from autodiver.skinny.skinny_characteristic import Skinny64Characteristic, Skinny128Characteristic
from autodiver.speck.speck_characteristic import Speck32Characteristic, Speck48Characteristic, Speck64Characteristic, Speck96Characteristic, Speck128Characteristic
from autodiver.speedy192.speedy192_model import Speedy192Characteristic
from autodiver.warp128.warp_characteristic import WarpCharacteristic

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAILS_DIR = REPO_ROOT / "trails"

CHARACTERISTIC_GLOBS: dict[type[DifferentialCharacteristic], str] = {
    AsconCharacteristic: "ascon/*.npz",
    Gift64Characteristic: "gift64/*.txt",
    Gift128Characteristic: "gift128/*.txt",
    Midori64Characteristic: "midori64/*.npz",
    Midori128Characteristic: "midori128/*.npz",
    PresentCharacteristic: "present/*.npz",
    Pyjamask96Characteristic: "pyjamask96/*.npz",
    RectangleCharacteristic: "rectangle/*.txt",
    Skinny64Characteristic: "skinny64/*.npz",
    Skinny128Characteristic: "skinny128/*.npz",
    Speck32Characteristic: "speck/speck32_*.npz",
    Speck48Characteristic: "speck/speck48_*.npz",
    Speck64Characteristic: "speck/speck64_*.npz",
    Speck96Characteristic: "speck/speck96_*.npz",
    Speck128Characteristic: "speck/speck128_*.npz",
    Speedy192Characteristic: "speedy192/*.txt",
    WarpCharacteristic: "warp/*.npz",
}


def _collect_trails() -> list[tuple[type[DifferentialCharacteristic], Path]]:
    trails = []
    for cls, glob in CHARACTERISTIC_GLOBS.items():
        paths = sorted(TRAILS_DIR.glob(glob))
        if not paths:
            warnings.warn(f"no characteristics found for {cls.__name__} (glob {glob!r})")
        for path in paths:
            trails.append((cls, path))
    return trails


_TRAILS = _collect_trails()


@pytest.mark.parametrize("characteristic_cls,characteristic_path", _TRAILS, ids=[path.stem for cls, path in _TRAILS])
def test_verify_linear_layer(
    characteristic_cls: type[DifferentialCharacteristic], characteristic_path: Path
):
    char = characteristic_cls.load(characteristic_path)
    char.verify_linear_layer()

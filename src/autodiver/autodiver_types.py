from __future__ import annotations
from pysat.card import EncType
from enum import Enum, unique
from dataclasses import dataclass
from sat_toolkit.formula import CNF

@unique
class ModelType(Enum):
    solution_set = 'solution-set'
    split_solution_set = 'split-solution-set'

class RoundMode(Enum):
    UP = 'up'
    DOWN = 'down'

class SearchMode(Enum):
    UPWARDS = 'upwards'
    BINARY = 'binary'

CARD_ENC_MAP = {
    name: value
    for name, value in vars(EncType).items()
    if not name.startswith("_") and isinstance(value, int)
}


@dataclass(slots=True)
class CharSearchParams:
    related_tweak: bool = False
    rounding_mode: RoundMode = RoundMode.DOWN
    searching_mode: SearchMode = SearchMode.BINARY
    log_prob_boundary: int | None = None
    card_enc: int = 8

class UnsatException(Exception):
    pass

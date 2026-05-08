from __future__ import annotations
from pysat.card import EncType
from enum import Enum, unique

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


class UnsatException(Exception):
    pass

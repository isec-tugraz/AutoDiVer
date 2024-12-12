from __future__ import annotations

from enum import Enum, unique

@unique
class ModelType(Enum):
    solution_set = 'solution-set'
    split_solution_set = 'split-solution-set'


class UnsatException(Exception):
    pass

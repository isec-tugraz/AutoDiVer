from __future__ import annotations
from collections.abc import Iterable
from numbers import Integral
from sat_toolkit.formula import CNF


class XorClause():
    def __init__(self, literals: Iterable[Integral]):
        self.literals = set()
        for literal in literals:
            self.add(literal)

    def xor_constant(self, constant: bool|int):
        if constant not in (0, 1):
            raise ValueError("Constant must be 0/1 or False/True")
        if len(self.literals) == 0:
            raise ValueError("Cannot xor constant with empty clause")
        if constant:
            lit = self.literals.pop()
            self.literals.add(-lit)

    def add(self, literal: Integral):
        if literal in self.literals:
            self.literals.remove(literal)
        else:
            self.literals.add(literal)

    def to_cnf(self):
        rhs = 0
        args = []
        for literal in self.literals:
            rhs ^= literal < 0
            args.append([abs(literal)])
        return CNF.create_xor(*args, rhs=rhs)

    def __ixor__(self, other: Integral|XorClause):
        if isinstance(other, XorClause):
            for literal in other.literals:
                self ^= literal
        assert isinstance(other, Integral)
        if other in self.literals:
            self.literals.remove(other)
        else:
            self.literals.add(other)
        return self

    def __xor__(self, other: Integral|XorClause):
        result = XorClause(self.literals)
        result ^= other
        return result

    def __repr__(self):
        return f"XorClause({self.literals!r})"

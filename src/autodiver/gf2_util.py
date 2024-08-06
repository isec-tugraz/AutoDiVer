from __future__ import annotations

import numpy as np
import random
from galois import GF2

def affine_hull(points: list[GF2]) -> AffineSpace:
    offset = points[0]
    vectors = GF2(points[1:])
    vectors += offset

    basis_matrix = vectors.row_space()

    return AffineSpace(offset, basis_matrix)


class AffineSpace:
    def __init__(self, offset: GF2, basis_matrix: GF2):
        if offset.ndim != 1:
            raise ValueError("Offset must be a vector")
        if basis_matrix.ndim != 2:
            raise ValueError("Basis matrix must be a matrix")
        if basis_matrix.shape[1] != offset.size:
            raise ValueError("size of offset vector must match number of columns in basis matrix")

        reduced_basis = basis_matrix.row_space()
        if len(basis_matrix) != len(reduced_basis):
            raise ValueError(f"Basis matrix ({basis_matrix.shape[0]}x{basis_matrix.shape[1]}) must be full rank (has rank {len(reduced_basis)})")

        self.offset = offset
        self.basis_matrix = basis_matrix

    def dimension(self) -> int:
        return len(self.basis_matrix)

    def element_size(self) -> int:
        assert len(self.offset) == len(self.basis_matrix[0])
        return len(self.offset)

    def __len__(self) -> int:
        return 2**self.dimension()

    def __iter__(self):
        dim = self.dimension()
        for i in range(2 ** dim):
            mul = GF2([(i >> j) & 1 for j in range(dim)])
            yield self.offset + mul @ self.basis_matrix

    def as_equation_system(self) -> tuple[GF2, GF2]:
        """
        Returns the affine space as a system of linear equations Ax = b

        Returns:
        tuple: (A, b)
        """

        # we now have an affine space for the possible keys:
        # K = offset + v * basis_matrix (for all v)
        # we can multiply with right_kern, the right kernel of basis_matrix
        # K * right_kern = offset * right_kern
        # right_kern.T * K = right_kern.T * offset

        right_kern = self.basis_matrix.T.left_null_space().T
        assert np.all(self.basis_matrix @ right_kern == 0)

        A = right_kern.T
        b = right_kern.T @ self.offset

        return A, b

    def random_sample(self) -> GF2:
        """
        Returns a random element from the affine space.
        """
        rnd = random.getrandbits(self.dimension())
        selector = GF2([(rnd >> i) & 1 for i in range(self.dimension())])
        return self.offset + selector @ self.basis_matrix


import pytest

import optimesh

from .helpers import assert_norm_equality
from .meshes import pacman, simple1


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [5.0, 2.1213203435596424, 1.0]),
        (pacman, [1919.3310978354305, 75.03937100433645, 5.0]),
    ],
)
def test_fixed_point(mesh, ref):
    X, cells = mesh()
    X, cells = optimesh.optimize_points_cells(X, cells, "laplace", 0.0, 10)
    assert_norm_equality(X, ref, 1.0e-12)

import numpy
import pytest

import optimesh

from .meshes import circle_random


def _rotate(X, theta, k):
    # <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>
    return (
        X * numpy.cos(theta)
        + numpy.cross(k, X) * numpy.sin(theta)
        + numpy.outer(numpy.einsum("ij,j->i", X, k), k) * (1.0 - numpy.cos(theta))
    )


@pytest.mark.parametrize(
    "method",
    [
        optimesh.cpt.fixed_point_uniform,
        optimesh.cpt.quasi_newton_uniform,
        #
        optimesh.cvt.quasi_newton_uniform_lloyd,
        lambda *args: optimesh.cvt.quasi_newton_uniform_lloyd(*args, omega=2.0),
        optimesh.cvt.quasi_newton_uniform_blocks,
        lambda *args: optimesh.cvt.quasi_newton_uniform_full(*args, omega=0.9),
        #
        optimesh.odt.fixed_point_uniform,
        optimesh.odt.nonlinear_optimization_uniform,
    ],
)
def test_rotated(method):
    X, cells = circle_random(40, 1.0)
    X = numpy.column_stack([X, numpy.zeros(X.shape[0])])

    # Apply a robust method first to avoid too crazy meshes.
    X, cells = optimesh.cpt.fixed_point_uniform(X, cells, 0.0, 2)

    X_orig = X.copy()
    cells_orig = cells.copy()

    # Create reference solution
    num_steps = 10
    X_ref, cells_ref = method(X, cells, 0.0, num_steps)

    # Create a rotated mesh
    theta = numpy.pi / 4
    k = numpy.array([1.0, 0.0, 0.0])
    X_rot = _rotate(X_orig, theta, k)
    cells_rot = cells_orig.copy()
    X2, cells2 = method(X_rot, cells_rot, 0.0, num_steps)
    # rotate back
    X2 = _rotate(X2, -theta, k)

    assert numpy.all(cells_ref == cells2)
    assert numpy.all(numpy.abs(X_ref - X2) < 1.0e-12)
    return


if __name__ == "__main__":
    test_rotated(optimesh.cpt.fixed_point_uniform)

# -*- coding: utf-8 -*-
#
import numpy
import pytest

import optimesh

from meshes import simple1, simple2, simple3, pacman, circle


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 5.0, 2.1213203435596424, 1.0),
        (simple2, 7.390123456790124, 2.804687217072868, 1.7),
        (simple3, 12.000533426212133, 3.9766966218492676, 2.0),
        (pacman, 1914.89271008783, 75.07101948912008, 5.0),
    ],
)
def test_fixed_point(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.odt.fixed_point_uniform(X, cells, 1.0e-3, 100, uniform_density=True)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - ref1) < tol * ref1
    assert abs(norm2 - ref2) < tol * ref2
    assert abs(normi - refi) < tol * refi
    return


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 4.999994919473657, 2.1213191460738456, 1.0),
        (simple2, 7.374076666666667, 2.8007819180622477, 1.7),
        (simple3, 12.000000734595783, 3.9828838201616144, 2.0),
        # (pacman, 1919.2497615803882, 75.226990639805, 5.0),
    ],
)
def test_nonlinear_optimization(mesh, ref1, ref2, refi):
    X, cells = mesh()

    # TODO remove
    X = X[:, :2]

    X, cells = optimesh.odt.nonlinear_optimization_uniform(X, cells, 1.0e-5, 100)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - ref1) < tol * ref1
    assert abs(norm2 - ref2) < tol * ref2
    assert abs(normi - refi) < tol * refi
    return


if __name__ == "__main__":
    X, cells = circle()
    X, cells = optimesh.odt.fixed_point(X, cells, 1.0e-3, 100)

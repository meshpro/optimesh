# -*- coding: utf-8 -*-
#
import numpy
import pytest

import optimesh

from meshes import simple1, pacman


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 4.985355657854027, 2.1179164560036154, 1.0),
        (pacman, 1939.1198108068188, 75.94965207932323, 5.0),
    ],
)
def test_simple_cvt(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.cvt.fixed_point_uniform(
        X, cells, 1.0e-2, 100, fcc_type="boundary", verbosity=2
    )

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

# -*- coding: utf-8 -*-
#
import numpy

import optimesh

from meshes import simple1, pacman


def test_simple_lloyd(max_num_steps=5):
    X, cells = simple1()

    X, cells = optimesh.lloyd(X, cells, 1.0e-2, 100, fcc_type="boundary", verbosity=2)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 4.985355657854027
    assert abs(norm1 - ref) < tol * ref
    ref = 2.1179164560036154
    assert abs(norm2 - ref) < tol * ref
    ref = 1.0
    assert abs(normi - ref) < tol * ref

    return


def test_pacman_lloyd(max_num_steps=1000):
    X, cells = pacman()

    X, _ = optimesh.lloyd(X, cells, 1.0e-2, 100, fcc_type="boundary", verbosity=1)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 1939.1198108068188
    assert abs(norm1 - ref) < tol * ref
    ref = 75.94965207932323
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref

    return


if __name__ == "__main__":
    # test_simple_lloyd(
    test_pacman_lloyd(max_num_steps=100)

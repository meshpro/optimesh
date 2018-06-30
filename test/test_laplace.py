# -*- coding: utf-8 -*-
#
import numpy

import meshio
import optimesh

from helpers import download_mesh


def test_simple(num_steps=10, output_filetype=None):
    X = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.4, 0.5, 0.0],
        ]
    )
    cells = numpy.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

    X, cells = optimesh.laplace(
        X, cells, 0.0, num_steps, output_filetype=output_filetype
    )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 5.0
    assert abs(norm1 - ref) < tol * ref
    ref = 2.1213203435596424
    assert abs(norm2 - ref) < tol * ref
    ref = 1.0
    assert abs(normi - ref) < tol * ref

    return


def test_pacman(num_steps=10, output_filetype=None):
    filename = download_mesh(
        "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
    )
    mesh = meshio.read(filename)

    X, cells = optimesh.laplace(
        mesh.points,
        mesh.cells["triangle"],
        0.0,
        num_steps,
        verbosity=1,
        output_filetype=output_filetype,
    )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 1917.9950540725958
    assert abs(norm1 - ref) < tol * ref
    ref = 74.99386491032608
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref

    return


if __name__ == "__main__":
    # test_laplace(
    # test_pacman(num_steps=100, output_filetype="png")
    test_pacman(num_steps=100, output_filetype=None)

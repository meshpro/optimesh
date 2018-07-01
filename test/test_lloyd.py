# -*- coding: utf-8 -*-
#
import numpy

import meshio
import optimesh

from helpers import download_mesh


def test_simple_lloyd(max_num_steps=5, output_filetype=None):
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

    submesh_bools = {0: numpy.ones(len(cells), dtype=bool)}

    X, cells = optimesh.lloyd_submesh(
        X,
        cells,
        1.0e-2,
        100,
        submesh_bools,
        skip_inhomogenous_submeshes=True,
        fcc_type="boundary",
        verbosity=2,
        output_filetype=output_filetype,
    )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - 4.9853556578540266) < tol
    assert abs(norm2 - 2.1179164560036154) < tol
    assert abs(normi - 1.0) < tol

    return


def test_pacman_lloyd(max_num_steps=1000, output_filetype=None):
    filename = download_mesh(
        "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
    )
    mesh = meshio.read(filename)

    submesh_bools = {0: numpy.ones(len(mesh.cells["triangle"]), dtype=bool)}

    X, cells = optimesh.lloyd_submesh(
        mesh.points,
        mesh.cells["triangle"],
        1.0e-2,
        100,
        submesh_bools,
        skip_inhomogenous_submeshes=False,
        fcc_type="boundary",
        flip_frequency=1,
        verbosity=1,
        output_filetype=output_filetype,
    )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    # assert abs(norm1 - 1944.49523751269) < tol
    # assert abs(norm2 - 76.097893244864181) < tol
    assert abs(norm1 - 1939.1198108068188) < tol
    assert abs(norm2 - 75.949652079323229) < tol
    assert abs(normi - 5.0) < tol

    return


if __name__ == "__main__":
    # test_simple_lloyd(
    test_pacman_lloyd(max_num_steps=100, output_filetype=None)

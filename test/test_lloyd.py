# -*- coding: utf-8 -*-
#
import numpy

import meshio
import voropy

from helpers import download_mesh


def test_simple_lloyd(max_steps=5, output_filetype=None):
    X = numpy.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.4, 0.5, 0.0],
        ])
    cells = numpy.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        ])

    submesh_bools = {0: numpy.ones(len(cells), dtype=bool)}

    X, cells = voropy.smoothing.lloyd_submesh(
        X, cells, submesh_bools,
        1.0e-2,
        skip_inhomogenous_submeshes=True,
        max_steps=max_steps,
        fcc_type='boundary',
        verbose=True,
        output_filetype=output_filetype
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


def test_pacman_lloyd(max_steps=1000, output_filetype=None):
    filename = download_mesh(
        'pacman.msh',
        '2da8ff96537f844a95a83abb48471b6a'
        )
    X, cells, _, _, _ = meshio.read(filename)

    submesh_bools = {0: numpy.ones(len(cells['triangle']), dtype=bool)}

    X, cells = voropy.smoothing.lloyd_submesh(
        X, cells['triangle'], submesh_bools,
        1.0e-2,
        skip_inhomogenous_submeshes=False,
        max_steps=max_steps,
        fcc_type='boundary',
        flip_frequency=1,
        verbose=False,
        output_filetype=output_filetype
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


if __name__ == '__main__':
    # test_pacman_lloyd(
    test_simple_lloyd(
        max_steps=100,
        output_filetype='png'
        )

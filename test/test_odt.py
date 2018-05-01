# -*- coding: utf-8 -*-
#
import numpy

import meshio
import optimesh

from helpers import download_mesh


def test_simple1():
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

    X, cells = optimesh.odt(X, cells, tol=1.0e-5)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 4.999994919473657
    assert abs(norm1 - ref) < tol * ref
    ref = 2.1213191460738456
    assert abs(norm2 - ref) < tol * ref
    ref = 1.0
    assert abs(normi - ref) < tol * ref

    return


def test_simple2():
    X = numpy.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.5, 0.0],
        [1.7, 0.5, 0.0],
        ])
    cells = numpy.array([
        [0, 1, 4],
        [1, 5, 4],
        [2, 4, 5],
        [2, 3, 4],
        [3, 0, 4],
        ])

    X, cells = optimesh.odt(X, cells)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 7.374076666666667
    assert abs(norm1 - ref) < tol * ref
    ref = 2.8007819180622477
    assert abs(norm2 - ref) < tol * ref
    ref = 1.7
    assert abs(normi - ref) < tol * ref

    return


def test_simple3():
    X = numpy.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.5, 0.0],
        [1.7, 0.5, 0.0],
        ])
    cells = numpy.array([
        [0, 1, 6],
        [1, 7, 6],
        [1, 2, 7],
        [2, 3, 7],
        [3, 4, 7],
        [4, 6, 7],
        [4, 5, 6],
        [5, 0, 6],
        ])

    X, cells = optimesh.odt(X, cells)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 12.000000734595783
    assert abs(norm1 - ref) < tol * ref
    ref = 3.9828838201616144
    assert abs(norm2 - ref) < tol * ref
    ref = 2.0
    assert abs(normi - ref) < tol * ref

    return


# def test_circle():
#     X, cells, _, _, _ = meshio.read('circle.vtk')
#     cells = cells['triangle']
#     X, cells = optimesh.odt(X, cells)
#     return


def test_pacman():
    filename = download_mesh(
        'pacman.msh',
        '601a51e53d573ff58bfec96aef790f0bb6c531a221fd7841693eaa20'
        )
    X, cells, _, _, _ = meshio.read(filename)

    X, cells = optimesh.odt(
        X, cells['triangle'],
        verbose=True,
        tol=1.0e-5
        )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 1918.8077833218122
    assert abs(norm1 - ref) < tol * ref
    ref = 75.21321080665695
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref

    return


if __name__ == '__main__':
    # test_simple1()
    # test_simple2()
    # test_simple3()
    # test_circle()
    test_pacman()

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

    X, cells = optimesh.odt_chen(X, cells, tol=1.0e-5)

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

    X, cells = optimesh.odt_chen(X, cells)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 7.374074074074074
    assert abs(norm1 - ref) < tol * ref
    ref = 2.8007812940925643
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

    X, cells = optimesh.odt_chen(X, cells)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 12.000268061419682
    assert abs(norm1 - ref) < tol * ref
    ref = 3.9829396222966804
    assert abs(norm2 - ref) < tol * ref
    ref = 2.0
    assert abs(normi - ref) < tol * ref

    return


def test_circle():
    import pygmsh
    geom = pygmsh.built_in.Geometry()

    geom.add_circle(
        [0.0, 0.0, 0.0],
        1.0,
        # 5.0e-3,
        1.0e-2,
        num_sections=4,
        # If compound==False, the section borders have to be points of the
        # discretization. If using a compound circle, they don't; gmsh can
        # choose by itself where to point the circle points.
        compound=True
        )
    X, cells, _, _, _ = pygmsh.generate_mesh(geom)

    X, cells = optimesh.odt_chen(
        X, cells['triangle'],
        verbose=True,
        tol=1.0e-3
        )
    return


def test_pacman():
    filename = download_mesh(
        'pacman.msh',
        '601a51e53d573ff58bfec96aef790f0bb6c531a221fd7841693eaa20'
        )
    X, cells, _, _, _ = meshio.read(filename)

    X, cells = optimesh.odt_chen(
        X, cells['triangle'],
        verbose=True,
        tol=1.0e-3
        )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-10
    ref = 1918.756560192194
    assert abs(norm1 - ref) < tol * ref
    ref = 75.21580844291586
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref

    return


if __name__ == '__main__':
    # test_simple1()
    # test_simple2()
    # test_simple3()
    test_circle()
    # test_pacman()

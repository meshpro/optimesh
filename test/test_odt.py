# -*- coding: utf-8 -*-
#
import os

import meshio
import numpy

import optimesh

from meshes import simple1, simple2, simple3, pacman


def test_simple1_fixed_point():
    X, cells = simple1()

    X, cells = optimesh.odt.fixed_point(X, cells, 1.0e-5, 100, uniform_density=True)

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


def test_simple2_fixed_point():
    X, cells = simple2()

    X, cells = optimesh.odt.fixed_point(X, cells, 1.0e-3, 100, uniform_density=True)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 7.390123456790124
    assert abs(norm1 - ref) < tol * ref
    ref = 2.804687217072868
    assert abs(norm2 - ref) < tol * ref
    ref = 1.7
    assert abs(normi - ref) < tol * ref
    return


def test_simple3_fixed_point():
    X, cells = simple3()

    X, cells = optimesh.odt.fixed_point(X, cells, 1.0e-3, 100, uniform_density=True)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 12.000533426212133
    assert abs(norm1 - ref) < tol * ref
    ref = 3.9766966218492676
    assert abs(norm2 - ref) < tol * ref
    ref = 2.0
    assert abs(normi - ref) < tol * ref
    return


def test_circle_fixed_point():
    filename = "circle.vtk"
    if not os.path.isfile(filename):
        import pygmsh

        geom = pygmsh.built_in.Geometry()
        geom.add_circle(
            [0.0, 0.0, 0.0],
            1.0,
            5.0e-3,
            num_sections=4,
            # If compound==False, the section borders have to be points of the
            # discretization. If using a compound circle, they don't; gmsh can
            # choose by itself where to point the circle points.
            compound=True,
        )
        X, cells, _, _, _ = pygmsh.generate_mesh(
            geom, fast_conversion=True, remove_faces=True
        )
        meshio.write_points_cells(filename, X, cells)

    mesh = meshio.read(filename)
    c = mesh.cells["triangle"].astype(numpy.int)

    X, cells = optimesh.odt.fixed_point(mesh.points, c, 1.0e-3, 100)
    return


def test_pacman_fixed_point():
    X, cells = pacman()

    X, _ = optimesh.odt.fixed_point(X, cells, 1.0e-3, 500, uniform_density=True)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-10
    ref = 1913.6793612758217
    assert abs(norm1 - ref) < tol * ref
    ref = 75.04142473590268
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref
    return


def test_simple1_nonlinear_optimization():
    # TODO use 3D coordinates, simple1
    X = numpy.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.4, 0.5]])
    cells = numpy.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

    X, cells = optimesh.odt.nonlinear_optimization(X, cells, 1.0e-5, 100)

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


def test_simple2_nonlinear_optimization():
    # TODO use three coordinates, simple2()
    X = numpy.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.7, 0.5], [1.7, 0.5]]
    )
    cells = numpy.array([[0, 1, 4], [1, 5, 4], [2, 4, 5], [2, 3, 4], [3, 0, 4]])

    X, cells = optimesh.odt.nonlinear_optimization(X, cells, 1.0e-5, 100)

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
    # TODO use simple3()
    X = numpy.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.7, 0.5],
            [1.7, 0.5],
        ]
    )
    cells = numpy.array(
        [
            [0, 1, 6],
            [1, 7, 6],
            [1, 2, 7],
            [2, 3, 7],
            [3, 4, 7],
            [4, 6, 7],
            [4, 5, 6],
            [5, 0, 6],
        ]
    )

    X, cells = optimesh.odt.nonlinear_optimization(X, cells, 1.0e-5, 100)

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


def test_pacman_nonlinear_optimization():
    X, cells = pacman()
    assert numpy.all(numpy.abs(X[:, 2]) < 1.0e-15)
    X = X[:, :2]

    X, cells = optimesh.odt.nonlinear_optimization(X, cells, 1.0e-5, 100)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-8
    ref = 1919.249752617539
    assert abs(norm1 - ref) < tol * ref
    ref = 75.22699025430875
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref
    return


def circle_nonlinear_optimization():
    filename = "circle.vtk"
    if not os.path.isfile(filename):
        import pygmsh

        geom = pygmsh.built_in.Geometry()
        geom.add_circle(
            [0.0, 0.0, 0.0],
            1.0,
            5.0e-3,
            # 1.0e-2,
            num_sections=4,
            # If compound==False, the section borders have to be points of the
            # discretization. If using a compound circle, they don't; gmsh can
            # choose by itself where to point the circle points.
            compound=True,
        )
        X, cells, _, _, _ = pygmsh.generate_mesh(
            geom, fast_conversion=True, remove_faces=True
        )
        meshio.write_points_cells(filename, X, cells)

    mesh = meshio.read(filename)

    # TODO remove this
    X = mesh.points[:, :2]

    c = mesh.cells["triangle"].astype(numpy.int)

    X, cells = optimesh.odt.nonlinear_optimization(
        X,
        c,
        # 3.0e-8,
        2.0e-8,
        100,
        verbosity=1,
    )
    return


if __name__ == "__main__":
    test_pacman_nonlinear_optimization()

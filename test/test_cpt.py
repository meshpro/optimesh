# -*- coding: utf-8 -*-
#
import os.path

import numpy

import meshio
import optimesh

from helpers import download_mesh


def test_simple1_energy():
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

    tol = 1.0e-12

    energy = optimesh.cpt.energy(X, cells, uniform_density=False)
    ref = 17.0 / 60.0
    assert abs(energy - ref) < tol * ref

    energy = optimesh.cpt.energy(X, cells, uniform_density=True)
    ref = 101.0 / 90.0
    assert abs(energy - ref) < tol * ref
    return


def test_pacman_energy():
    filename = download_mesh(
        "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
    )
    mesh = meshio.read(filename)

    X = mesh.points
    cells = mesh.cells["triangle"]

    tol = 1.0e-12

    energy = optimesh.cpt.energy(X, cells, uniform_density=False)
    ref = 7.320400634147646
    assert abs(energy - ref) < tol * ref

    energy = optimesh.cpt.energy(X, cells, uniform_density=True)
    ref = 78.8877511729188
    assert abs(energy - ref) < tol * ref
    return


def test_simple1_fixed_point():
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

    X, cells = optimesh.cpt.fixed_point(X, cells, 1.0e-12, 100, uniform_density=True)

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
    X = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.5, 0.0],
            [1.7, 0.5, 0.0],
        ]
    )
    cells = numpy.array([[0, 1, 4], [1, 5, 4], [2, 4, 5], [2, 3, 4], [3, 0, 4]])

    X, cells = optimesh.cpt.fixed_point(X, cells, 1.0e-3, 100, uniform_density=True)

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
    X = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.5, 0.0],
            [1.7, 0.5, 0.0],
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

    X, cells = optimesh.cpt.fixed_point(X, cells, 1.0e-3, 100, uniform_density=True)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 12.000175966581171
    assert abs(norm1 - ref) < tol * ref
    ref = 3.9766184197352734
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
    c = mesh.cells["triangle"].astype(numpy.int)

    X, cells = optimesh.cpt.fixed_point(mesh.points, c, 1.0e-3, 100)
    return


def test_pacman_fixed_point():
    filename = download_mesh(
        "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
    )
    mesh = meshio.read(filename)

    X, _ = optimesh.cpt.fixed_point(
        mesh.points, mesh.cells["triangle"], 1.0e-3, 100, uniform_density=True
    )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-10
    ref = 1905.563081121851
    assert abs(norm1 - ref) < tol * ref
    ref = 74.72133890269812
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref

    return


def test_simple1_linear_solve():
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

    X, cells = optimesh.cpt.linear_solve(X, cells, 1.0e-12, 100, uniform_density=True)

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


def test_simple2_linear_solve():
    X = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.5, 0.0],
            [1.7, 0.5, 0.0],
        ]
    )
    cells = numpy.array([[0, 1, 4], [1, 5, 4], [2, 4, 5], [2, 3, 4], [3, 0, 4]])

    X, cells = optimesh.cpt.linear_solve(X, cells, 1.0e-3, 100, uniform_density=True)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 7.44
    assert abs(norm1 - ref) < tol * ref
    ref = 2.8173746644704534
    assert abs(norm2 - ref) < tol * ref
    ref = 1.7
    assert abs(normi - ref) < tol * ref

    return


def test_simple3_linear_solve():
    X = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.5, 0.0],
            [1.7, 0.5, 0.0],
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

    X, cells = optimesh.cpt.linear_solve(X, cells, 1.0e-3, 100, uniform_density=True)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 12.0
    assert abs(norm1 - ref) < tol * ref
    ref = 3.9651257511234395
    assert abs(norm2 - ref) < tol * ref
    ref = 2.0
    assert abs(normi - ref) < tol * ref

    return


def test_circle_linear_solve():
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
    c = mesh.cells["triangle"].astype(numpy.int)

    X, cells = optimesh.cpt.linear_solve(mesh.points, c, 1.0e-3, 100)
    return


def test_pacman_linear_solve():
    filename = download_mesh(
        "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
    )
    mesh = meshio.read(filename)

    X, _ = optimesh.cpt.linear_solve(
        mesh.points, mesh.cells["triangle"], 1.0e-3, 100, uniform_density=True
    )

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-10
    ref = 1861.1845669965835
    assert abs(norm1 - ref) < tol * ref
    ref = 73.12639677867615
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref

    return


if __name__ == "__main__":
    test_pacman_fixed_point()

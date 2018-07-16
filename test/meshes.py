# -*- coding: utf-8 -*-
#
import os.path

import meshio
import numpy
from scipy.spatial import Delaunay

from helpers import download_mesh


def simple0():
    #
    #  3___________2
    #  |\_   2   _/|
    #  |  \_   _/  |
    #  | 3  \4/  1 |
    #  |   _/ \_   |
    #  | _/     \_ |
    #  |/    0    \|
    #  0-----------1
    #
    X = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    cells = numpy.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return X, cells


def simple1():
    #
    #  3___________2
    #  |\_   2   _/|
    #  |  \_   _/  |
    #  | 3  \4/  1 |
    #  |   _/ \_   |
    #  | _/     \_ |
    #  |/    0    \|
    #  0-----------1
    #
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
    return X, cells


def simple2():
    #
    #  3___________2
    #  |\_   3   _/ \_
    #  |  \_   _/  2  \_
    #  | 4  \4/_________\5
    #  |   _/ \_       _/
    #  | _/     \_ 1 _/
    #  |/    0    \ /
    #  0-----------1
    #
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
    return X, cells


def simple3():
    #
    #  5___________4___________3
    #  |\_   6   _/ \_   4   _/|
    #  |  \_   _/  5  \_   _/  |
    #  | 7  \6/_________\7/  3 |
    #  |   _/ \_       _/ \_   |
    #  | _/     \_ 1 _/  2  \_ |
    #  |/    0    \ /         \|
    #  0-----------1-----------2
    #
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
    return X, cells


def pacman():
    filename = download_mesh(
        "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
    )
    mesh = meshio.read(filename)
    return mesh.points, mesh.cells["triangle"]


def circle():
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
    return mesh.points, c


def circle_random():
    n = 40
    boundary_pts = numpy.array(
        [
            [numpy.cos(2 * numpy.pi * k / n), numpy.sin(2 * numpy.pi * k / n)]
            for k in range(n)
        ]
    )

    # generate random points in circle; <http://mathworld.wolfram.com/DiskPointPicking.html>
    numpy.random.seed(0)
    m = 200
    r = numpy.random.rand(m)
    alpha = 2 * numpy.pi * numpy.random.rand(m)

    interior_pts = numpy.column_stack(
        [numpy.sqrt(r) * numpy.cos(alpha), numpy.sqrt(r) * numpy.sin(alpha)]
    )

    pts = numpy.concatenate([boundary_pts, interior_pts])

    tri = Delaunay(pts)
    pts = numpy.column_stack([pts[:, 0], pts[:, 1], numpy.zeros(pts.shape[0])])
    return pts, tri.simplices

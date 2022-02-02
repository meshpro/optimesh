import pathlib

import meshio
import meshplex
import numpy as np
from scipy.spatial import Delaunay

this_dir = pathlib.Path(__file__).resolve().parent


def simple_line():
    X = np.array([0.0, 0.1, 0.4, 1.0])
    cells = np.array([[0, 1], [1, 2], [2, 3]])
    return meshplex.Mesh(X, cells)


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
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return meshplex.MeshTri(X, cells)


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
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.4, 0.5]])
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return meshplex.MeshTri(X, cells)


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
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.7, 0.5], [1.7, 0.5]]
    )
    cells = np.array([[0, 1, 4], [1, 5, 4], [2, 4, 5], [2, 3, 4], [3, 0, 4]])
    return meshplex.MeshTri(X, cells)


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
    X = np.array(
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
    cells = np.array(
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
    return meshplex.MeshTri(X, cells)


def pacman():
    mesh = meshio.read(this_dir / "meshes" / "pacman.vtk")
    return meshplex.MeshTri(mesh.points[:, :2], mesh.get_cells_type("triangle"))


def circle_gmsh():
    mesh = meshio.read(this_dir / "meshes" / "circle.vtk")
    c = mesh.get_cells_type("triangle")
    return mesh.points[:, :2], c


def _compute_num_boundary_points(total_num_points):
    # The number of boundary points, the total number of points, and the number of cells
    # are connected by two equations (the second of which is approximate).
    #
    # Euler:
    # 2 * num_points - num_boundary_edges - 2 = num_cells
    #
    # edge_length = 2 * np.pi / num_boundary_points
    # tri_area = np.sqrt(3) / 4 * edge_length ** 2
    # num_cells = int(np.pi / tri_area)
    #
    # num_boundary_points = num_boundary_edges
    #
    # Hence:
    # 2 * num_points =
    # num_boundary_points + 2 + np.pi / (np.sqrt(3) / 4 * (2 * np.pi / num_boundary_points) ** 2)
    #
    # We need to solve
    #
    # + num_boundary_points ** 2
    # + (sqrt(3) * pi) * num_boundary_points
    # + (2 - 2 * num_points) * (sqrt(3) * pi)
    # = 0
    #
    # for the number of boundary points.
    sqrt3_pi = np.sqrt(3) * np.pi
    num_boundary_points = -sqrt3_pi / 2 + np.sqrt(
        3 / 4 * np.pi**2 - (2 - 2 * total_num_points) * sqrt3_pi
    )
    return num_boundary_points


def circle_gmsh2():
    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # target_edge_length = 2 * np.pi / _compute_num_boundary_points(num_points)
    # geom.add_circle(
    #     [0.0, 0.0, 0.0], 1.0, lcar=target_edge_length, num_sections=4, compound=True
    # )
    # mesh = pygmsh.generate_mesh(geom, remove_lower_dim_cells=True, verbose=False)
    # mesh.write("out.vtk")

    mesh = meshio.read(this_dir / "meshes" / "circle-small.vtk")
    c = mesh.get_cells_type("triangle")
    return mesh.points[:, :2], c


def circle_random(n, radius):
    k = np.arange(n)
    boundary_pts = radius * np.column_stack(
        [np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)]
    )

    # Compute the number of interior points such that all triangles can be somewhat
    # equilateral.
    edge_length = 2 * np.pi * radius / n
    domain_area = np.pi - n * (radius**2 / 2 * (edge_length - np.sin(edge_length)))
    cell_area = np.sqrt(3) / 4 * edge_length**2
    target_num_cells = domain_area / cell_area
    # Euler:
    # 2 * num_points - num_boundary_edges - 2 = num_cells
    # <=>
    # num_interior_points ~= 0.5 * (num_cells + num_boundary_edges) + 1 - num_boundary_points
    m = int(0.5 * (target_num_cells + n) + 1 - n)

    # generate random points in circle; <https://mathworld.wolfram.com/DiskPointPicking.html>
    np.random.seed(0)
    r = np.random.rand(m)
    alpha = 2 * np.pi * np.random.rand(m)

    interior_pts = np.column_stack(
        [np.sqrt(r) * np.cos(alpha), np.sqrt(r) * np.sin(alpha)]
    )

    pts = np.concatenate([boundary_pts, interior_pts])

    tri = Delaunay(pts)

    # Make sure there are exactly `n` boundary points
    mesh = meshplex.MeshTri(pts, tri.simplices)
    assert np.sum(mesh.is_boundary_point) == n

    return pts, tri.simplices


def circle_random2(n, radius, seed=0):
    """Boundary points are random, too."""
    # generate random points in circle; <https://mathworld.wolfram.com/DiskPointPicking.html>
    np.random.seed(seed)
    r = np.random.rand(n)
    alpha = 2 * np.pi * np.random.rand(n)

    pts = np.column_stack([np.sqrt(r) * np.cos(alpha), np.sqrt(r) * np.sin(alpha)])
    tri = Delaunay(pts)
    # Make sure there are exactly `n` boundary points
    mesh = meshplex.MeshTri(pts, tri.simplices)
    # inflate the mesh such that the boundary points average around the radius
    boundary_pts = pts[mesh.is_boundary_point]
    dist = np.sqrt(np.einsum("ij,ij->i", boundary_pts, boundary_pts))
    avg_dist = np.sum(dist) / len(dist)
    mesh.points = pts / avg_dist
    # boundary_pts = pts[mesh.is_boundary_point]
    # dist = np.sqrt(np.einsum("ij,ij->i", boundary_pts, boundary_pts))
    # avg_dist = np.sum(dist) / len(dist)
    # print(avg_dist)

    # now move all boundary points to the circle
    # bpts = pts[mesh.is_boundary_point]
    # pts[mesh.is_boundary_point] = (
    #     bpts.T / np.sqrt(np.einsum("ij,ij->i", bpts, bpts))
    # ).T
    # bpts = pts[mesh.is_boundary_point]
    # print(np.sqrt(np.einsum("ij,ij->i", bpts, bpts)))
    # mesh = meshplex.MeshTri(pts, tri.simplices)
    # mesh.show()

    return pts, tri.simplices


def circle_rotated():
    pts, cells = circle_random()
    # <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>
    theta = np.pi / 4
    k = np.array([1.0, 0.0, 0.0])
    pts = (
        pts * np.cos(theta)
        + np.cross(k, pts) * np.sin(theta)
        + np.outer(np.einsum("ij,j->i", pts, k), k) * (1.0 - np.cos(theta))
    )
    meshio.write_points_cells("out.vtk", pts, {"triangle": cells})
    return pts, cells

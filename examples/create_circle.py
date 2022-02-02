import numpy as np
from scipy.spatial import Delaunay


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


def create_random_circle(num_boundary_points, num_interior_points, radius):
    n = num_boundary_points
    k = 2 * np.pi * np.arange(n) / n
    boundary_pts = radius * np.array([np.cos(k), np.sin(k)])

    # # Compute the number of interior points such that all triangles can be somewhat
    # # equilateral.
    # edge_length = 2 * np.pi * radius / n
    # domain_area = np.pi - n * (
    #     radius ** 2 / 2 * (edge_length - np.sin(edge_length))
    # )
    # cell_area = np.sqrt(3) / 4 * edge_length ** 2
    # target_num_cells = domain_area / cell_area
    # # Euler:
    # # 2 * num_points - num_boundary_edges - 2 = num_cells
    # # <=>
    # # num_interior_points ~= 0.5 * (num_cells + num_boundary_edges) + 1 - num_boundary_points
    # m = int(0.5 * (target_num_cells + n) + 1 - n)

    # Generate random points in circle;
    # <https://mathworld.wolfram.com/DiskPointPicking.html>.
    seed = 0
    while True:
        np.random.seed(seed)
        r = np.random.rand(num_interior_points)
        alpha = 2 * np.pi * np.random.rand(num_interior_points)
        interior_pts = np.sqrt(r) * np.array([np.cos(alpha), np.sin(alpha)])
        # Check if no interior point will be on the boundary
        is_any_outside = False
        for k in range(n):
            v = boundary_pts[:, k] - boundary_pts[:, k - 1]
            q = (interior_pts.T - boundary_pts[:, k - 1]).T
            is_any_outside = np.any(v[0] * q[1] <= v[1] * q[0])
            if is_any_outside:
                break

        if not is_any_outside:
            break

        seed += 1

    pts = np.concatenate([boundary_pts.T, interior_pts.T])

    tri = Delaunay(pts)
    # pts = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0])])
    return pts, tri.simplices


def random(num_points):
    num_boundary_points = int(_compute_num_boundary_points(num_points))
    num_interior_points = num_points - num_boundary_points
    pts, cells = create_random_circle(
        num_boundary_points, num_interior_points, radius=1.0
    )
    # import matplotlib.pyplot as plt
    # plt.plot(pts[:, 0], pts[:, 1], ".")
    # plt.axis("equal")
    # plt.show()
    # exit(1)
    return pts, cells


def gmsh(num_points):
    import pygmsh

    geom = pygmsh.built_in.Geometry()
    target_edge_length = 2 * np.pi / _compute_num_boundary_points(num_points)
    geom.add_circle(
        [0.0, 0.0, 0.0], 1.0, lcar=target_edge_length, num_sections=4, compound=True
    )
    mesh = pygmsh.generate_mesh(geom, remove_lower_dim_cells=True, verbose=False)
    return mesh.points[:, :2], mesh.cells[0].data


def dmsh(target_num_points):
    import dmsh

    print("target num points", target_num_points)

    est_num_boundary_points = _compute_num_boundary_points(target_num_points)
    # est_num_boundary_points = 100
    target_edge_length = 2 * np.pi / est_num_boundary_points
    print(target_edge_length)
    print("est num boundary", est_num_boundary_points)
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, target_edge_length)
    print("num points", X.shape[0])

    import meshplex

    mesh = meshplex.MeshTri(X, cells)
    print("num boundary points", sum(mesh.is_boundary_point))
    # exit(1)
    return X, cells


if __name__ == "__main__":
    random(5)
    # gmsh()

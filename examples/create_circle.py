import numpy
from scipy.spatial import Delaunay

import meshio
from meshplex import MeshTri


def create_random_circle(n, radius, seed=None):
    k = numpy.arange(n)
    boundary_pts = radius * numpy.column_stack(
        [numpy.cos(2 * numpy.pi * k / n), numpy.sin(2 * numpy.pi * k / n)]
    )

    # Compute the number of interior nodes such that all triangles can be somewhat
    # equilateral.
    edge_length = 2 * numpy.pi * radius / n
    domain_area = numpy.pi - n * (
        radius ** 2 / 2 * (edge_length - numpy.sin(edge_length))
    )
    cell_area = numpy.sqrt(3) / 4 * edge_length ** 2
    target_num_cells = domain_area / cell_area
    # Euler:
    # 2 * num_points - num_boundary_edges - 2 = num_cells
    # <=>
    # num_interior_points ~= 0.5 * (num_cells + num_boundary_edges) + 1 - num_boundary_points
    m = int(0.5 * (target_num_cells + n) + 1 - n)

    # Generate random points in circle;
    # <http://mathworld.wolfram.com/DiskPointPicking.html>.
    # Choose the seed such that the fully smoothened mesh has no random boundary points.
    if seed is not None:
        numpy.random.seed(seed)
    r = numpy.random.rand(m)
    alpha = 2 * numpy.pi * numpy.random.rand(m)

    interior_pts = numpy.column_stack(
        [numpy.sqrt(r) * numpy.cos(alpha), numpy.sqrt(r) * numpy.sin(alpha)]
    )

    pts = numpy.concatenate([boundary_pts, interior_pts])

    tri = Delaunay(pts)
    # pts = numpy.column_stack([pts[:, 0], pts[:, 1], numpy.zeros(pts.shape[0])])
    return pts, tri.simplices


def random():
    n = 40
    pts, cells = create_random_circle(n, radius=1.0, seed=0)
    assert numpy.sum(MeshTri(pts, cells).is_boundary_node) == n

    meshio.write_points_cells("circle.xdmf", pts, {"triangle": cells})
    return


def gmsh():
    import pygmsh

    geom = pygmsh.built_in.Geometry()
    geom.add_circle([0.0, 0.0, 0.0], 1.0, lcar=1.0e-1, num_sections=4, compound=True)
    mesh = pygmsh.generate_mesh(geom)
    meshio.write("circle-gmsh.vtk", mesh)
    return


if __name__ == "__main__":
    random()
    # gmsh()

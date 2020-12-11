import meshplex
import numpy
import pytest
from scipy.spatial import Delaunay

import optimesh

from .helpers import assert_norm_equality
from .meshes import pacman, simple1


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [4.9863354526224510, 2.1181412069258942, 1.0]),
        (pacman, [1.9378493850318487e03, 7.5989333945604329e01, 5.0]),
    ],
)
def test_cvt_lloyd(mesh, ref):
    X, cells = mesh()
    # X, cells = optimesh.cvt.quasi_newton_uniform_lloyd(
    #     X, cells, 1.0e-2, 100, verbose=False
    # )
    m = meshplex.MeshTri(X, cells)
    optimesh.optimize(m, "Lloyd", 1.0e-2, 100, verbose=False)
    assert_norm_equality(m.points, ref, 1.0e-12)

    # try the other way of calling optimesh
    X, c = mesh()
    X, _ = optimesh.optimize_points_cells(X, c, "lloyd", 1.0e-2, 100)
    assert_norm_equality(X, ref, 1.0e-12)


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [4.9959407761650168e00, 2.1203672449514870e00, 1.0]),
        (pacman, [1.9369945166933908e03, 7.5977272076992293e01, 5.0]),
    ],
)
def test_cvt_lloyd_overrelaxed(mesh, ref):
    X, cells = mesh()
    m = meshplex.MeshTri(X, cells)
    optimesh.optimize(m, "Lloyd", 1.0e-2, 100, omega=2.0)
    assert_norm_equality(m.points, ref, 1.0e-12)


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [4.9957677170205690e00, 2.1203267741647247e00, 1.0]),
        (pacman, [1.9368767965291165e03, 7.5956311011221615e01, 5.0]),
    ],
)
def test_cvt_qnb(mesh, ref):
    X, cells = mesh()
    m = meshplex.MeshTri(X, cells)
    optimesh.optimize(m, "CVT (block-diagonal)", 1.0e-2, 100)
    assert_norm_equality(m.points, ref, 1.0e-10)


def test_cvt_qnb_boundary(n=10):
    X, cells = create_random_circle(n=n, radius=1.0)

    def boundary_step(x):
        x0 = [0.0, 0.0]
        r = 1.0
        # simply project onto the circle
        y = (x.T - x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
        return ((y / r * r).T + x0).T

    mesh = meshplex.MeshTri(X, cells)
    optimesh.optimize(mesh, "Lloyd", 1.0e-2, 100, boundary_step=boundary_step)

    # X, cells = optimesh.cvt.quasi_newton_uniform_lloyd(
    #     X, cells, 1.0e-2, 100, boundary_step=boundary_step
    # )
    # X, cells = optimesh.cvt.quasi_newton_uniform_blocks(
    #     X, cells, 1.0e-2, 100, boundary=Circle()
    # )

    mesh.show()

    # Assert that we're dealing with the mesh we expect.
    # assert_norm_equality(X, [ref1, ref2, refi], 1.0e-12)


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 4.9971490009329251e00, 2.1206501666066013e00, 1.0),
        (pacman, 1.9384734362432773e03, 7.5992449567867354e01, 5.0),
    ],
)
def test_cvt_qnf(mesh, ref1, ref2, refi):
    X, cells = mesh()
    X, cells = optimesh.optimize_points_cells(
        X, cells, "cvt (full)", 1.0e-2, 100, omega=0.9
    )

    import meshplex

    mesh = meshplex.MeshTri(X, cells)
    mesh.show()

    # Assert that we're dealing with the mesh we expect.
    assert_norm_equality(X, [ref1, ref2, refi], 1.0e-12)


def create_random_circle(n, radius, seed=0):
    k = numpy.arange(n)
    boundary_pts = radius * numpy.column_stack(
        [numpy.cos(2 * numpy.pi * k / n), numpy.sin(2 * numpy.pi * k / n)]
    )

    # Compute the number of interior points such that all triangles can be somewhat
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


# This test iterates over a few meshes that produce weird sitations that did have the
# methods choke. Mostly bugs in GhostedMesh.
@pytest.mark.parametrize("seed", [0, 4, 20])
def test_for_breakdown(seed):
    numpy.random.seed(seed)

    n = numpy.random.randint(10, 20)
    pts, cells = create_random_circle(n=n, radius=1.0)

    optimesh.optimize_points_cells(
        pts, cells, "lloyd", omega=1.0, tol=1.0e-10, max_num_steps=10
    )


if __name__ == "__main__":
    test_cvt_qnb_boundary(50)

import copy

import meshplex
import numpy as np
import pytest
from scipy.spatial import Delaunay

import optimesh

from . import meshes
from .helpers import assert_norm_equality

pacman = meshes.pacman()
simple1 = meshes.simple1()
simple_line = meshes.simple_line()


@pytest.mark.parametrize(
    "mesh, num_steps, ref",
    [
        (simple_line, 1, [1.55, 1.0972978173677372, 1.0]),
        (simple_line, 100, [1.9550946819363164e00, 1.2295013498442391e00, 1.0]),
        #
        (simple1, 1, [4.9319444444444445e00, 2.1063181153582713e00, 1.0]),
        (simple1, 100, [4.9863354526224510, 2.1181412069258942, 1.0]),
        # We're adding relatively many tests here. The reason is that even small changes
        # in meshplex, e.g., in the computation of the circumcenters, can build up
        # across a CVT iteration and lead to differences that aren't so small. The
        # sequence of tests is makes sure that the difference builds up step-by-step and
        # isn't a sudden break.
        (pacman, 1, [1.9449825885691200e03, 7.6122084669586002e01, 5.0]),
        (pacman, 2, [1.9446726479253102e03, 7.6115000143782524e01, 5.0]),
        (pacman, 10, [1.9424088268502351e03, 7.6063446601225976e01, 5.0]),
        (pacman, 20, [1.9407096235482659e03, 7.6028721177100564e01, 5.0]),
        (pacman, 30, [1.9397254043011189e03, 7.6011552957849773e01, 5.0]),
        (pacman, 40, [1.9391902386060749e03, 7.6005991941058554e01, 5.0]),
        (pacman, 50, [1.9387458681835806e03, 7.6000274906909084e01, 5.0]),
        (pacman, 75, [1.9382955570646300e03, 7.5996522030844588e01, 5.0]),
        (pacman, 100, [1.9378463822717290e03, 7.5989210861590919e01, 5.0]),
    ],
)
def test_cvt_lloyd(mesh, num_steps, ref):
    print(num_steps)
    m = copy.deepcopy(mesh)
    optimesh.optimize(m, "Lloyd", 1.0e-2, num_steps, verbose=False)
    assert_norm_equality(m.points, ref, 1.0e-12)

    # try the other way of calling optimesh
    X, c = mesh.points.copy(), mesh.cells("points").copy()
    X, _ = optimesh.optimize_points_cells(X, c, "lloyd", 1.0e-2, num_steps)
    assert_norm_equality(X, ref, 1.0e-12)


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [4.9959407761650168e00, 2.1203672449514870e00, 1.0]),
        (pacman, [1.9366587758940354e03, 7.5962435325397195e01, 5.0]),
    ],
)
def test_cvt_lloyd_overrelaxed(mesh, ref):
    m = copy.deepcopy(mesh)
    optimesh.optimize(m, "Lloyd", 1.0e-2, 100, omega=2.0)
    assert_norm_equality(m.points, ref, 1.0e-12)


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [4.9957677170205690e00, 2.1203267741647247e00, 1.0]),
        (pacman, [1.9368767962543961e03, 7.5956311040257475e01, 5.0]),
    ],
)
def test_cvt_qnb(mesh, ref):
    m = copy.deepcopy(mesh)
    optimesh.optimize(m, "CVT (block-diagonal)", 1.0e-2, 100)
    assert_norm_equality(m.points, ref, 1.0e-9)


def test_cvt_qnb_boundary(n=10):
    X, cells = create_random_circle(n=n, radius=1.0)

    def boundary_step(x):
        x0 = [0.0, 0.0]
        R = 1.0
        # simply project onto the circle
        y = (x.T - x0).T
        r = np.sqrt(np.einsum("ij,ij->j", y, y))
        return ((y / r * R).T + x0).T

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
    "mesh, ref",
    [
        (simple1, [4.9971490009329251e00, 2.1206501666066013e00, 1.0]),
        (pacman, [1.9385249442149425e03, 7.5995141991060208e01, 5.0]),
    ],
)
def test_cvt_qnf(mesh, ref):
    m = copy.deepcopy(mesh)
    optimesh.optimize(m, "cvt (full)", 1.0e-2, 100, omega=0.9)
    m.show()
    # Assert that we're dealing with the mesh we expect.
    assert_norm_equality(m.points, ref, 1.0e-4)


def create_random_circle(n, radius, seed=0):
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

    # Generate random points in circle;
    # <https://mathworld.wolfram.com/DiskPointPicking.html>.
    # Choose the seed such that the fully smoothened mesh has no random boundary points.
    if seed is not None:
        np.random.seed(seed)
    r = np.random.rand(m)
    alpha = 2 * np.pi * np.random.rand(m)

    interior_pts = np.column_stack(
        [np.sqrt(r) * np.cos(alpha), np.sqrt(r) * np.sin(alpha)]
    )

    pts = np.concatenate([boundary_pts, interior_pts])

    tri = Delaunay(pts)
    # pts = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0])])
    return pts, tri.simplices


# This test iterates over a few meshes that produce weird situations that did have the
# methods choke. Mostly bugs in GhostedMesh.
@pytest.mark.parametrize("seed", [0, 4, 20])
def test_for_breakdown(seed):
    np.random.seed(seed)

    n = np.random.randint(10, 20)
    pts, cells = create_random_circle(n=n, radius=1.0)

    optimesh.optimize_points_cells(
        pts, cells, "lloyd", omega=1.0, tol=1.0e-10, max_num_steps=10
    )


if __name__ == "__main__":
    test_cvt_qnb_boundary(50)

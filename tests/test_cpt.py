import numpy as np
import pytest
import quadpy
from meshplex import MeshTri

import optimesh
from optimesh.cpt.quasi_newton import _jac_uniform

from . import meshes
from .helpers import assert_norm_equality

simple0 = meshes.simple0()
simple1 = meshes.simple1()
simple2 = meshes.simple2()
simple3 = meshes.simple3()
simple_line = meshes.simple_line()
pacman = meshes.pacman()


def _energy_uniform_per_point(X, cells):
    """The CPT mesh energy is defined as

        sum_i E_i,
        E_i = 1/(d+1) * sum int_{omega_i} ||x - x_i||^2 rho(x) dx,

    see Chen-Holst. This method gives the E_i and  assumes uniform density, rho(x) = 1.
    """
    mesh = MeshTri(X, cells)

    star_integrals = np.zeros(mesh.points.shape[0])
    # Python loop over the cells... slow!
    for cell in mesh.cells("points"):
        for idx in cell:
            xi = mesh.points[idx]
            tri = mesh.points[cell]
            # Get a scheme of order 2
            scheme = quadpy.t2.get_good_scheme(2)
            val = scheme.integrate(
                lambda x: np.einsum("ij,ij->i", x.T - xi, x.T - xi), tri
            )
            star_integrals[idx] += val

    dim = 2
    return star_integrals / (dim + 1)


def _energy_uniform(X, cells):
    return np.sum(_energy_uniform_per_point(X, cells))


@pytest.mark.parametrize(
    "mesh, ref",
    [(simple0, 5.0 / 18.0), (simple1, 17.0 / 60.0), (pacman, 7.320400634147646)],
)
def test_energy(mesh, ref):
    X, cells = mesh.points, mesh.cells("points")
    energy = _energy_uniform(X, cells)
    assert abs(energy - ref) < 1.0e-12 * ref


def test_simple1_jac():
    X, cells = simple1.points, simple1.cells("points")
    # First assert that the Jacobian at interior points coincides with the finite
    # difference computed for the energy component from that point. Note that the
    # contribution from all other points is disregarded here, just like in the
    # definition of the Jacobian of Chen-Holst; it's only an approximation after all.
    jac = _jac_uniform(X, cells)
    for j in [0, 1]:
        eps = 1.0e-7
        x0 = X.copy()
        x1 = X.copy()
        x0[4, j] -= eps
        x1[4, j] += eps
        f1 = _energy_uniform_per_point(x1, cells)
        f0 = _energy_uniform_per_point(x0, cells)
        dE = (f1 - f0) / (2 * eps)
        assert abs(dE[4] - jac[4, j]) < 1.0e-9


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [0.96, 0.3262279745178587, 29.0 / 225.0]),
        (pacman, [12.35078985438217, 0.5420691555930099, 0.10101179397867549]),
    ],
)
def test_jac(mesh, ref):
    X, cells = mesh.points, mesh.cells("points")
    jac = _jac_uniform(X, cells)
    assert_norm_equality(jac, ref, 1.0e-12)


@pytest.mark.parametrize(
    "method, mesh, ref",
    [
        ("cpt (fixed-point)", simple_line, [2.0, 1.2472191289241747e00, 1.0]),
        #
        ("cpt (fixed-point)", simple1, [5.0, 2.1213203435596424, 1.0]),
        ("cpt (fixed-point)", simple2, [7.390123456790124, 2.804687217072868, 1.7]),
        ("cpt (fixed-point)", simple3, [12.0, 3.9765648779799356, 2.0]),
        ("cpt (fixed-point)", pacman, [1901.5304112865315, 74.62452940437535, 5.0]),
        #
        ("cpt (quasi-newton)", simple1, [5.0, 2.1213203435596424, 1.0]),
        ("cpt (quasi-newton)", simple2, [7.390123456790124, 2.804687217072868, 1.7]),
        ("cpt (quasi-newton)", simple3, [12.0, 3.976564877979913, 2.0]),
        ("cpt (quasi-newton)", pacman, [1900.910794007578, 74.58866209782154, 5.0]),
    ],
)
def test_methods(method, mesh, ref):
    X_in, cells_in = mesh.points, mesh.cells("points")

    # X_before = X_in.copy()
    # cells_before = cells_in.copy()

    X, _ = optimesh.optimize_points_cells(X_in, cells_in, method, 1.0e-12, 100)

    # assert np.all(cells_in == cells_before)
    # assert np.all(np.abs(X_in == X_before) < 1.0e-15)

    # Test if we're dealing with the mesh we expect.
    assert_norm_equality(X, ref, 1.0e-12)


@pytest.mark.parametrize(
    "mesh, ref",
    [
        (simple1, [5.0, 2.1213203435596424, 1.0]),
        (pacman, [1864.2406342781524, 73.19722600427883, 5.0]),
    ],
)
def test_density_preserving(mesh, ref):
    X, cells = mesh.points, mesh.cells("points")
    X, cells = optimesh.optimize_points_cells(X, cells, "cpt (linear solve)", 0.0, 10)
    assert_norm_equality(X, ref, 1.0e-12)


def test_circle():
    def boundary_step(x):
        x0 = [0.0, 0.0]
        R = 1.0
        # simply project onto the circle
        y = (x.T - x0).T
        r = np.sqrt(np.einsum("ij,ij->j", y, y))
        return ((y / r * R).T + x0).T

    X, cells = meshes.circle_random2(150, 1.0)
    X, cells = optimesh.optimize_points_cells(
        X, cells, "cpt (fixed-point)", 1.0e-3, 100, boundary_step=boundary_step
    )

    mesh = MeshTri(X, cells)
    mesh.show()


if __name__ == "__main__":
    test_circle()

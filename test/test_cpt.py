import numpy
import pytest
import quadpy
from meshplex import MeshTri

import optimesh
from optimesh.cpt.quasi_newton import _jac_uniform

from .meshes import circle_random2, pacman, simple0, simple1, simple2, simple3


def _energy_uniform_per_point(X, cells):
    """The CPT mesh energy is defined as

        sum_i E_i,
        E_i = 1/(d+1) * sum int_{omega_i} ||x - x_i||^2 rho(x) dx,

    see Chen-Holst. This method gives the E_i and  assumes uniform density, rho(x) = 1.
    """
    mesh = MeshTri(X, cells)

    star_integrals = numpy.zeros(mesh.points.shape[0])
    # Python loop over the cells... slow!
    for cell in mesh.cells["points"]:
        for idx in cell:
            xi = mesh.points[idx]
            tri = mesh.points[cell]
            # Get a scheme of order 2
            scheme = quadpy.t2.get_good_scheme(2)
            val = scheme.integrate(
                lambda x: numpy.einsum("ij,ij->i", x.T - xi, x.T - xi), tri
            )
            star_integrals[idx] += val

    dim = 2
    return star_integrals / (dim + 1)


def _energy_uniform(X, cells):
    return numpy.sum(_energy_uniform_per_point(X, cells))


@pytest.mark.parametrize(
    "mesh, ref",
    [(simple0, 5.0 / 18.0), (simple1, 17.0 / 60.0), (pacman, 7.320400634147646)],
)
def test_energy(mesh, ref):
    X, cells = mesh()
    energy = _energy_uniform(X, cells)
    assert abs(energy - ref) < 1.0e-12 * ref


def test_simple1_jac():
    X, cells = simple1()
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
    "mesh, ref1, ref2, refi",
    [
        (simple1, 0.96, 0.3262279745178587, 29.0 / 225.0),
        (pacman, 12.35078985438217, 0.5420691555930099, 0.10101179397867549),
    ],
)
def test_jac(mesh, ref1, ref2, refi):
    X, cells = mesh()

    jac = _jac_uniform(X, cells)

    nc = jac.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - ref1) < tol * ref1
    assert abs(norm2 - ref2) < tol * ref2
    assert abs(normi - refi) < tol * refi


@pytest.mark.parametrize(
    "method, mesh, ref1, ref2, refi",
    [
        ("cpt (fixed-point)", simple1, 5.0, 2.1213203435596424, 1.0),
        ("cpt (fixed-point)", simple2, 7.390123456790124, 2.804687217072868, 1.7),
        ("cpt (fixed-point)", simple3, 12.0, 3.9765648779799356, 2.0),
        ("cpt (fixed-point)", pacman, 1901.5304112865315, 74.62452940437535, 5.0),
        #
        ("cpt (quasi-newton)", simple1, 5.0, 2.1213203435596424, 1.0),
        ("cpt (quasi-newton)", simple2, 7.390123456790124, 2.804687217072868, 1.7),
        ("cpt (quasi-newton)", simple3, 12.0, 3.976564877979913, 2.0),
        ("cpt (quasi-newton)", pacman, 1900.910794007578, 74.58866209782154, 5.0),
    ],
)
def test_methods(method, mesh, ref1, ref2, refi):
    X_in, cells_in = mesh()

    # X_before = X_in.copy()
    # cells_before = cells_in.copy()

    X, _ = optimesh.optimize_points_cells(X_in, cells_in, method, 1.0e-12, 100)

    # assert numpy.all(cells_in == cells_before)
    # assert numpy.all(numpy.abs(X_in == X_before) < 1.0e-15)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - ref1) < tol * ref1
    assert abs(norm2 - ref2) < tol * ref2
    assert abs(normi - refi) < tol * refi


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 5.0, 2.1213203435596424, 1.0),
        (pacman, 1864.2406342781524, 73.19722600427883, 5.0),
    ],
)
def test_density_preserving(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.optimize_points_cells(X, cells, "cpt (linear solve)", 0.0, 10)

    # Test if we're dealing with the mesh we expect.
    nc = X.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - ref1) < tol * ref1
    assert abs(norm2 - ref2) < tol * ref2
    assert abs(normi - refi) < tol * refi


def test_circle():
    def boundary_step(x):
        x0 = [0.0, 0.0]
        r = 1.0
        # simply project onto the circle
        y = (x.T - x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
        return ((y / r * r).T + x0).T

    X, cells = circle_random2(150, 1.0)
    X, cells = optimesh.optimize_points_cells(
        X, cells, "cpt (fixed-point)", 1.0e-3, 100, boundary_step=boundary_step
    )


if __name__ == "__main__":
    test_circle()

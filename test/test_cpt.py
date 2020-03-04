import numpy
import pytest

from meshes import pacman, simple0, simple1, simple2, simple3
from optimesh import cpt


@pytest.mark.parametrize(
    "mesh, ref",
    [(simple0, 5.0 / 18.0), (simple1, 17.0 / 60.0), (pacman, 7.320400634147646)],
)
def test_energy(mesh, ref):
    X, cells = mesh()
    energy = cpt.energy_uniform(X, cells)
    assert abs(energy - ref) < 1.0e-12 * ref
    return


def test_simple1_jac():
    X, cells = simple1()

    # First assert that the Jacobian at interior points coincides with the finite
    # difference computed for the energy component from that point. Note that the
    # contribution from all other points is disregarded here, just like in the
    # definition of the Jacobian of Chen-Holst; it's only an approximation after all.
    jac = cpt.jac_uniform(X, cells)
    for j in [0, 1]:
        eps = 1.0e-7
        x0 = X.copy()
        x1 = X.copy()
        x0[4, j] -= eps
        x1[4, j] += eps
        f1 = cpt._energy_uniform_per_node(x1, cells)
        f0 = cpt._energy_uniform_per_node(x0, cells)
        dE = (f1 - f0) / (2 * eps)
        assert abs(dE[4] - jac[4, j]) < 1.0e-10

    return


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 0.96, 0.3262279745178587, 29.0 / 225.0),
        (pacman, 12.35078985438217, 0.5420691555930099, 0.10101179397867549),
    ],
)
def test_jac(mesh, ref1, ref2, refi):
    X, cells = mesh()

    jac = cpt.jac_uniform(X, cells)

    nc = jac.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - ref1) < tol * ref1
    assert abs(norm2 - ref2) < tol * ref2
    assert abs(normi - refi) < tol * refi
    return


@pytest.mark.parametrize(
    "method, mesh, ref1, ref2, refi",
    [
        (cpt.fixed_point_uniform, simple1, 5.0, 2.1213203435596424, 1.0),
        (cpt.fixed_point_uniform, simple2, 7.390123456790124, 2.804687217072868, 1.7),
        (cpt.fixed_point_uniform, simple3, 12.0, 3.9765648779799356, 2.0),
        (cpt.fixed_point_uniform, pacman, 1901.5304112865315, 74.62452940437535, 5.0),
        #
        (cpt.quasi_newton_uniform, simple1, 5.0, 2.1213203435596424, 1.0),
        (cpt.quasi_newton_uniform, simple2, 7.390123456790124, 2.804687217072868, 1.7),
        (cpt.quasi_newton_uniform, simple3, 12.0, 3.976564877979913, 2.0),
        (cpt.quasi_newton_uniform, pacman, 1900.910794007578, 74.58866209782154, 5.0),
    ],
)
def test_methods(method, mesh, ref1, ref2, refi):
    X_in, cells_in = mesh()

    # X_before = X_in.copy()
    # cells_before = cells_in.copy()

    X, cells = method(X_in, cells_in, 1.0e-12, 100)

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
    return


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 5.0, 2.1213203435596424, 1.0),
        (pacman, 1864.2406342781524, 73.19722600427883, 5.0),
    ],
)
def test_density_preserving(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = cpt.linear_solve_density_preserving(X, cells, 0.0, 10)

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
    class Circle:
        def __init__(self):
            self.x0 = [0.0, 0.0]
            self.r = 1.0

        def boundary_step(self, x):
            # simply project onto the circle
            y = (x.T - self.x0).T
            r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
            return ((y / r * self.r).T + self.x0).T

    from meshes import circle_random2

    X, cells = circle_random2(150, 1.0)
    X, cells = cpt.fixed_point_uniform(X, cells, 1.0e-3, 100, boundary=Circle())


if __name__ == "__main__":
    test_circle()

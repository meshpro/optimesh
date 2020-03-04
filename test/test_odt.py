import numpy
import pytest

import optimesh
from meshes import pacman, simple1, simple2, simple3


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 5.0, 2.1213203435596424, 1.0),
        (simple2, 7.390123456790124, 2.804687217072868, 1.7),
        (simple3, 12.00095727371816, 3.9768056113618786, 2.0),
        (pacman, 1913.7750600561587, 75.0497156713882, 5.0),
    ],
)
def test_fixed_point(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.odt.fixed_point_uniform(
        X, cells, 1.0e-3, 100, uniform_density=True
    )

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
        (simple2, 1991.0 / 270.0, 2.8007812940925643, 1.7),
        (simple3, 12.000001546277293, 3.9828845062967257, 2.0),
        # (pacman, 1919.2497615803882, 75.226990639805, 5.0),
    ],
)
def test_nonlinear_optimization(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.odt.nonlinear_optimization_uniform(X, cells, 1.0e-5, 100)

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

    from meshes import circle_gmsh2

    # ODT can't handle the random circle; some cells too flat near the boundary lead to
    # a breakdown.
    # X, cells = circle_random2(150, 1.0, seed=1)
    X, cells = circle_gmsh2(100)
    X, cells = optimesh.odt.fixed_point_uniform(
        X, cells, 1.0e-3, 100, boundary=Circle()
    )


if __name__ == "__main__":
    test_circle()

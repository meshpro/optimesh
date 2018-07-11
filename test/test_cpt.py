# -*- coding: utf-8 -*-
#
import numpy
import pytest

import optimesh

from meshes import simple1, simple2, simple3, pacman


@pytest.mark.parametrize("mesh, ref", [
    (simple1, 17.0 / 60.0),
    (pacman, 7.320400634147646),
])
def test_energy(mesh, ref):
    X, cells = mesh()
    energy = optimesh.cpt.energy_uniform(X, cells)
    assert abs(energy - ref) < 1.0e-12 * ref
    return


# def test_simple1_jac():
#     X, cells = _get_simple1()
#
#     jac = optimesh.cpt.jac_uniform(X, cells)
#
#     print(jac)
#     exit(1)
#     return


# def test_pacman_jac():
#     filename = download_mesh(
#         "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
#     )
#     mesh = meshio.read(filename)
#
#     X = mesh.points
#     cells = mesh.cells["triangle"]
#
#     tol = 1.0e-12
#
#     energy = optimesh.cpt.energy(X, cells, uniform_density=False)
#     ref = 7.320400634147646
#     assert abs(energy - ref) < tol * ref
#
#     energy = optimesh.cpt.energy(X, cells, uniform_density=True)
#     ref = 78.8877511729188
#     assert abs(energy - ref) < tol * ref
#     return


@pytest.mark.parametrize("mesh, ref1, ref2, refi", [
    (simple1, 5.0, 2.1213203435596424, 1.0),
    (simple2, 7.390123456790124, 2.804687217072868, 1.7),
    (simple3, 12.0, 3.9765648779799356, 2.0),
    (pacman, 1903.6345096485093, 74.6604068632378, 5.0),
])
def test_fixed_point(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.cpt.fixed_point_uniform(X, cells, 1.0e-12, 100)

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


@pytest.mark.parametrize("mesh, ref1, ref2, refi", [
    (simple1, 5.0, 2.1213203435596424, 1.0),
    (simple2, 7.44, 2.8173746644704534, 1.7),
    (simple3, 12.0, 3.9651257511234395, 2.0),
    (pacman, 1861.1845669965835, 73.12639677151657, 5.0),
])
def test_linear_solve(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.cpt.density_preserving(X, cells, 1.0e-12, 100)

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


# if __name__ == "__main__":
#     from meshes import circle
#     test_fixed_point()
#     X, cells = circle()
#     X, cells = optimesh.cpt.density_preserving(X, cells, 1.0e-3, 100)
#     X, cells = optimesh.cpt.fixed_point_uniform(X, cells, 1.0e-3, 100)

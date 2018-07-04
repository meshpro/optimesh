# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
import fastfunc

from meshplex import MeshTri

from .helpers import print_stats, energy


def odt(X, cells, tol, max_num_steps, verbosity=1, step_filename_format=None):
    """Optimal Delaunay Triangulation smoothing.

    This method minimizes the energy

        E = int_Omega |u_l(x) - u(x)| rho(x) dx

    where u(x) = ||x||^2, u_l is its piecewise linear nodal interpolation and
    rho is the density. Since u(x) is convex, u_l >= u everywhere and

        u_l(x) = sum_i phi_i(x) u(x_i)

    where phi_i is the hat function at x_i. With rho(x)=1, this gives

        E = int_Omega sum_i phi_i(x) u(x_i) - u(x)
          = 1/(d+1) sum_i ||x_i||^2 |omega_i| - int_Omega ||x||^2

    where d is the spatial dimension and omega_i is the star of x_i (the set of
    all simplices containing x_i).
    """
    import scipy.optimize

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    mesh = MeshTri(X, cells, flat_cell_correction=None)

    if step_filename_format:
        mesh.save(
            step_filename_format.format(0), show_centroids=False, show_coedges=False
        )

    if verbosity > 0:
        print("Before:")
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)

    mesh.mark_boundary()

    is_interior_node = numpy.logical_not(mesh.is_boundary_node)

    def f(x):
        interior_coords = x.reshape(-1, 2)
        coords = X.copy()
        coords[is_interior_node] = interior_coords
        mesh.update_node_coordinates(coords)
        return energy(mesh, uniform_density=True)

    # TODO put f and jac together
    def jac(x):
        interior_coords = x.reshape(-1, 2)
        coords = X.copy()
        coords[is_interior_node] = interior_coords

        mesh.update_node_coordinates(coords)

        grad = numpy.zeros(coords.shape)
        cc = mesh.get_cell_circumcenters()
        for i in range(3):
            mcn = mesh.cells["nodes"][:, i]
            fastfunc.add.at(grad, mcn, ((coords[mcn] - cc).T * mesh.cell_volumes).T)
        gdim = 2
        grad *= 2 / (gdim + 1)
        return grad[is_interior_node, :2].flatten()

    def flip_delaunay(x):
        flip_delaunay.step += 1
        # Flip the edges
        interior_coords = x.reshape(-1, 2)
        coords = X.copy()
        coords[is_interior_node] = interior_coords
        mesh.update_node_coordinates(coords)
        mesh.flip_until_delaunay()

        if step_filename_format:
            mesh.save(
                step_filename_format.format(flip_delaunay.step),
                show_centroids=False,
                show_coedges=False,
            )
        if verbosity > 1:
            print("\nStep {}:".format(flip_delaunay.step))
            print_stats(mesh, extra_cols=["energy: {}".format(f(x))])

        # mesh.show()
        # exit(1)
        return

    flip_delaunay.step = 0

    x0 = X[is_interior_node, :2].flatten()

    out = scipy.optimize.minimize(
        f,
        x0,
        jac=jac,
        method="CG",
        # method='newton-cg',
        tol=tol,
        callback=flip_delaunay,
        options={"maxiter": max_num_steps},
    )
    # Don't assert out.success; max_num_steps may be reached, that's fine.

    # One last edge flip
    interior_coords = out.x.reshape(-1, 2)
    coords = X.copy()
    coords[is_interior_node] = interior_coords
    mesh.update_node_coordinates(coords)
    mesh.flip_until_delaunay()

    if verbosity > 0:
        print("\nFinal ({} steps):".format(out.nit))
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)
        print()

    return mesh.node_coords, mesh.cells["nodes"]

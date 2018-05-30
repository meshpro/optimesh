# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
import fastfunc
import voropy
from voropy.mesh_tri import MeshTri

from .helpers import gather_stats, print_stats


def odt(X, cells, verbose=False, tol=1.0e-5):
    '''Optimal Delaunay Triangulation smoothing.

    This method minimized the energy

        E = int_Omega |u_l(x) - u(x)| rho(x) dx

    where u(x) = ||x||^2, u_l is its piecewise linear nodal interpolation and
    rho is the density. Since u(x) is convex, u_l >= u everywhere and

        u_l(x) = sum_i phi_i(x) u(x_i)

    where phi_i is the hat function at x_i. With rho(x)=1, this gives

        E = int_Omega sum_i phi_i(x) u(x_i) - u(x)
          = 1/(d+1) sum_i ||x_i||^2 |omega_i| - int_Omega ||x||^2

    where d is the spatial dimension and omega_i is the star of x_i (the set of
    all simplices containing x_i).
    '''
    import scipy.optimize
    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    mesh = MeshTri(X, cells, flat_cell_correction=None)
    initial_stats = gather_stats(mesh)

    mesh.mark_boundary()

    is_interior_node = numpy.logical_not(mesh.is_boundary_node)

    # flat triangles
    gdim = 2

    def f(x):
        interior_coords = x.reshape(-1, 2)
        coords = X.copy()
        coords[is_interior_node] = interior_coords
        mesh.update_node_coordinates(coords)

        # E~ = 1/(d+1) sum_i ||x_i||^2 |omega_i|
        # This also adds the values on
        star_volume = numpy.zeros(X.shape[0])
        for i in range(3):
            fastfunc.add.at(
                star_volume, mesh.cells['nodes'][:, i], mesh.cell_volumes
                )
        x2 = numpy.einsum('ij,ij->i', mesh.node_coords, mesh.node_coords)
        out = 1/(gdim+1) * numpy.dot(star_volume, x2)
        return out

    # TODO put f and jac together
    def jac(x):
        interior_coords = x.reshape(-1, 2)
        coords = X.copy()
        coords[is_interior_node] = interior_coords

        mesh.update_node_coordinates(coords)

        grad = numpy.zeros(coords.shape)
        cc = mesh.get_cell_circumcenters()
        for i in range(3):
            mcn = mesh.cells['nodes'][:, i]
            fastfunc.add.at(
                grad,
                mcn,
                ((coords[mcn] - cc).T * mesh.cell_volumes).T
                )
        grad *= 2 / (gdim+1)
        return grad[is_interior_node, :2].flatten()

    def flip_delaunay(x):
        flip_delaunay.step += 1
        # Flip the edges
        interior_coords = x.reshape(-1, 2)
        coords = X.copy()
        coords[is_interior_node] = interior_coords
        mesh.update_node_coordinates(coords)
        mesh.flip_until_delaunay()

        if verbose:
            print('\nstep: {}'.format(flip_delaunay.step))
            print_stats([gather_stats(mesh)])

        # mesh.show()
        # exit(1)
        return
    flip_delaunay.step = 0

    x0 = X[is_interior_node, :2].flatten()

    out = scipy.optimize.minimize(
        f, x0,
        jac=jac,
        method='CG',
        tol=tol,
        callback=flip_delaunay
        )
    assert out.success, out.message

    interior_coords = out.x.reshape(-1, 2)
    coords = X.copy()
    coords[is_interior_node] = interior_coords
    mesh.update_node_coordinates(coords)
    mesh.flip_until_delaunay()

    if verbose:
        print('\nBefore:' + 35*' ' + 'After:')
        print_stats([
            initial_stats,
            gather_stats(mesh),
            ])

    return mesh.node_coords, mesh.cells['nodes']

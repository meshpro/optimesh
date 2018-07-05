# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
from meshplex import MeshTri

from .helpers import print_stats, energy


def lloyd(
    X,
    cells,
    tol,
    max_num_steps,
    fcc_type="boundary",
    verbosity=1,
    step_filename_format=None,
):
    # flat mesh
    if X.shape[1] == 3:
        assert numpy.all(numpy.abs(X[:, 2]) < 1.0e-15)
        X = X[:, :2]

    original_X = X.copy()

    # create mesh data structure
    mesh = MeshTri(X, cells, flat_cell_correction=fcc_type)
    mesh.flip_until_delaunay()

    if step_filename_format:
        mesh.save(
            step_filename_format.format(0),
            show_centroids=False,
            show_coedges=False,
            show_axes=False,
            nondelaunay_edge_color="k",
        )

    if verbosity > 0:
        print("\nBefore:")
        extra_cols = ["energy: {:.5e}".format(energy(mesh, uniform_density=True))]
        print_stats(mesh, extra_cols=extra_cols)

    k = 0
    while True:
        k += 1

        # move interior points into centroids
        new_points = mesh.get_control_volume_centroids()[mesh.is_interior_node]

        original_orient = mesh.get_signed_tri_areas() > 0.0
        original_coords = mesh.node_coords[mesh.is_interior_node]

        # Step unless the orientation of any cell changes.
        alpha = 1.0
        while True:
            xnew = (1 - alpha) * original_coords + alpha * new_points
            # Preserve boundary nodes
            original_X[mesh.is_interior_node] = xnew
            # A new mesh is created in every step. Ugh. We do that since meshplex
            # doesn't have update_node_coordinates with flat_cell_correction.
            mesh = MeshTri(
                original_X, mesh.cells["nodes"], flat_cell_correction=fcc_type
            )
            mesh.mark_boundary()
            # mesh.update_node_coordinates(xnew)
            new_orient = mesh.get_signed_tri_areas() > 0.0
            if numpy.all(original_orient == new_orient):
                break
            alpha /= 2

        mesh.flip_until_delaunay()

        if step_filename_format:
            mesh.save(
                step_filename_format.format(k),
                show_centroids=False,
                show_coedges=False,
                show_axes=False,
                nondelaunay_edge_color="k",
            )

        # Abort the loop if the update is small
        diff = mesh.node_coords[mesh.is_interior_node] - original_coords
        if numpy.all(numpy.einsum("ij,ij->i", diff, diff) < tol ** 2):
            break

        if k >= max_num_steps:
            break

        if verbosity > 1:
            print("\nstep {}:".format(k))
            print_stats(mesh)

    if verbosity > 0:
        print("\nFinal ({} steps):".format(k))
        extra_cols = ["energy: {:.5e}".format(energy(mesh, uniform_density=True))]
        print_stats(mesh, extra_cols=extra_cols)
        print()

    return mesh.node_coords, mesh.cells["nodes"]

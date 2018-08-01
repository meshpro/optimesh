# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
from meshplex import MeshTri

from .helpers import print_stats


def fixed_point_uniform(
    X, cells, tol, max_num_steps, verbosity=1, callback=None, step_filename_format=None
):
    """Lloyd's algorithm.
    """
    # flat mesh
    if X.shape[1] == 3:
        assert numpy.all(numpy.abs(X[:, 2]) < 1.0e-15)
        X = X[:, :2]

    original_X = X.copy()

    # create mesh data structure
    fcc_type = "boundary"
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
        print_stats(mesh)

    k = 0
    if callback:
        callback(k, mesh)

    while True:
        k += 1

        # move interior points into centroids
        new_points = mesh.control_volume_centroids[mesh.is_interior_node]

        original_orient = mesh.signed_tri_areas > 0.0
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
            # mesh.update_node_coordinates(xnew)
            new_orient = mesh.signed_tri_areas > 0.0
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

        if callback:
            callback(k, mesh)

        if k >= max_num_steps:
            break

        if verbosity > 1:
            print("\nstep {}:".format(k))
            print_stats(mesh)

    if verbosity > 0:
        print("\nFinal ({} steps):".format(k))
        print_stats(mesh)
        print()

    return mesh.node_coords, mesh.cells["nodes"]

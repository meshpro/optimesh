# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
from meshplex import MeshTri

from .helpers import extract_submesh_entities, print_stats, energy


def lloyd(
    X,
    cells,
    tol,
    max_num_steps,
    fcc_type="full",
    verbosity=1,
    step_filename_format=None,
):
    # flat mesh
    if X.shape[1] == 3:
        assert numpy.all(numpy.abs(X[:, 2]) < 1.0e-15)
        X = X[:, :2]

    # create mesh data structure
    mesh = MeshTri(X, cells, flat_cell_correction=fcc_type)
    mesh.flip_until_delaunay()

    boundary_verts = mesh.get_boundary_vertices()

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
        new_points = mesh.get_control_volume_centroids()
        new_points[boundary_verts] = mesh.node_coords[boundary_verts]

        original_orient = mesh.get_signed_tri_areas() > 0.0
        original_coords = mesh.node_coords.copy()

        # Step unless the orientation of any cell changes.
        alpha = 1.0
        while True:
            xnew = (1 - alpha) * original_coords + alpha * new_points
            # Preserve boundary nodes
            xnew[mesh.is_boundary_node] = original_coords[mesh.is_boundary_node]
            # A new mesh is created in every step. Ugh. We do that since meshplex
            # doesn't have update_node_coordinates with flat_cell_correction.
            mesh = MeshTri(xnew, mesh.cells["nodes"], flat_cell_correction=fcc_type)
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
        diff = new_points - mesh.node_coords
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


def lloyd_submesh(X, cells, tol, max_num_steps, submeshes, **kwargs):
    # perform lloyd on each submesh separately
    for cell_in_submesh in submeshes.values():
        submesh_X, submesh_cells, submesh_verts = extract_submesh_entities(
            X, cells, cell_in_submesh
        )

        # perform lloyd smoothing
        X_out, cells_out = lloyd(submesh_X, submesh_cells, tol, max_num_steps, **kwargs)

        # write the points and cells back
        X[submesh_verts, :2] = X_out
        cells[cell_in_submesh] = submesh_verts[cells_out]

    return X, cells

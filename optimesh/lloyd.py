# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
from voropy.mesh_tri import MeshTri

from .helpers import (
    extract_submesh_entities,
    get_boundary_edge_ratio,
    sit_in_plane,
    gather_stats,
    print_stats,
    write,
)


# pylint: disable=too-many-arguments,too-many-locals
def lloyd(
    X,
    cells,
    tol,
    max_steps=10000,
    fcc_type="full",
    flip_frequency=0,
    verbose=True,
    output_filetype=None,
):
    # flat mesh
    assert sit_in_plane(X)

    # create mesh data structure
    mesh = MeshTri(X, cells, flat_cell_correction=fcc_type)

    boundary_verts = mesh.get_boundary_vertices()

    max_move = tol + 1

    if verbose:
        print("\nstep: {}".format(0))
        print_stats(*gather_stats(mesh))

    next_flip_at = 0
    flip_skip = 1
    for k in range(max_steps):
        if max_move < tol:
            break
        if output_filetype:
            write(mesh, "lloyd", output_filetype, k)

        if k == next_flip_at:
            is_flipped_del = mesh.flip_until_delaunay()
            # mesh, is_flipped_six = flip_for_six(mesh)
            # is_flipped = numpy.logical_or(is_flipped_del, is_flipped_six)
            is_flipped = is_flipped_del
            if flip_frequency > 0:
                # fixed flip frequency
                flip_skip = flip_frequency
            else:
                # If the mesh needed flipping, flip again next time. Otherwise
                # double the interval.
                if is_flipped:
                    flip_skip = 1
                else:
                    flip_skip *= 2
            next_flip_at = k + flip_skip

        # move interior points into centroids
        new_points = mesh.get_control_volume_centroids()
        new_points[boundary_verts] = mesh.node_coords[boundary_verts]
        diff = new_points - mesh.node_coords
        max_move = numpy.sqrt(numpy.max(numpy.sum(diff * diff, axis=1)))

        mesh = MeshTri(new_points, mesh.cells["nodes"], flat_cell_correction=fcc_type)
        # mesh.update_node_coordinates(new_points)

        if verbose:
            print("\nstep: {}".format(0))
            print("  maximum move: {:15e}".format(max_move))
            print_stats(*gather_stats(mesh))

    # Flip one last time.
    mesh.flip_until_delaunay()
    # mesh, is_flipped_six = flip_for_six(mesh)

    if output_filetype:
        write(mesh, "lloyd", output_filetype, max_steps)

    return mesh.node_coords, mesh.cells["nodes"]


def lloyd_submesh(X, cells, submeshes, tol, skip_inhomogenous_submeshes=True, **kwargs):
    # perform lloyd on each submesh separately
    for cell_in_submesh in submeshes.values():
        submesh_X, submesh_cells, submesh_verts = extract_submesh_entities(
            X, cells, cell_in_submesh
        )

        if skip_inhomogenous_submeshes:
            # Since we don't have access to the density field here, voropy's
            # Lloyd smoothing will always make all cells roughly equally large.
            # This is inappropriate if the mesh is meant to be inhomogeneous,
            # e.g., if there are boundary layers. As a heuristic for
            # inhomogenous meshes, check the lengths of the longest and the
            # shortest boundary edge. If they are roughtly equal, perform Lloyd
            # smoothing.
            ratio = get_boundary_edge_ratio(submesh_X, submesh_cells)
            if ratio > 1.5:
                print(
                    (
                        4 * " "
                        + "Subdomain boundary inhomogeneous "
                        + "(edge length ratio {:1.3f}). Skipping."
                    ).format(ratio)
                )
                continue

        # perform lloyd smoothing
        X_out, cells_out = lloyd(submesh_X, submesh_cells, tol, **kwargs)

        # write the points and cells back
        X[submesh_verts] = X_out
        cells[cell_in_submesh] = submesh_verts[cells_out]

    return X, cells

# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
from meshplex import MeshTri

from .helpers import write, print_stats, energy


def laplace(X, cells, tol, max_num_steps, verbosity=0, output_filetype=None):
    """Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    """
    # TODO bring back?
    # flat mesh
    # assert sit_in_plane(X)

    # create mesh data structure
    mesh = MeshTri(X, cells, flat_cell_correction=None)

    boundary_verts = mesh.get_boundary_vertices()

    if verbosity > 0:
        print("Before:")
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols)

    for k in range(max_num_steps):
        if output_filetype:
            write(mesh, "laplace", output_filetype, k)

        mesh.flip_until_delaunay()

        # move interior points into average of their neighbors
        # <https://stackoverflow.com/a/43096495/353337>
        # num_neighbors = numpy.bincount(mesh.edges['nodes'].flat)
        #
        num_neighbors = numpy.zeros(mesh.node_coords.shape[0], dtype=int)
        new_points = numpy.zeros(mesh.node_coords.shape)
        for edge in mesh.edges["nodes"]:
            num_neighbors[edge[0]] += 1
            num_neighbors[edge[1]] += 1
            new_points[edge[0]] += mesh.node_coords[edge[1]]
            new_points[edge[1]] += mesh.node_coords[edge[0]]
        new_points = (new_points.T / num_neighbors).T

        # Keep the boundary vertices in place
        new_points[boundary_verts] = mesh.node_coords[boundary_verts]

        diff = new_points - mesh.node_coords
        max_move = numpy.sqrt(numpy.max(numpy.einsum("ij,ij->i", diff, diff)))

        mesh = MeshTri(new_points, mesh.cells["nodes"], flat_cell_correction=None)

        if verbosity > 1:
            print("\nStep {}:".format(k + 1))
            print_stats(
                mesh,
                extra_cols=["  maximum move: {:.5e}".format(max_move)]
            )

        if max_move < tol:
            break

    if verbosity == 1:
        print("\nFinal ({} steps):".format(k + 1))
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)
        print()

    # Flip one last time.
    mesh.flip_until_delaunay()

    if output_filetype:
        write(mesh, "laplace", output_filetype, k + 1)

    return mesh.node_coords, mesh.cells["nodes"]

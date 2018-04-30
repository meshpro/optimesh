# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
from voropy.mesh_tri import MeshTri

from .helpers import (
    sit_in_plane, gather_stats, write, flip_until_delaunay, print_stats
    )


def laplace(X, cells, num_steps, verbose=True, output_filetype=None):
    '''Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    '''
    # flat mesh
    assert sit_in_plane(X)

    # create mesh data structure
    mesh = MeshTri(X, cells, flat_cell_correction=None)

    boundary_verts = mesh.get_boundary_vertices()

    initial_stats = gather_stats(mesh)

    for k in range(num_steps):
        print(k)
        if output_filetype:
            write(mesh, output_filetype, k)

        mesh, _ = flip_until_delaunay(mesh)

        # move interior points into average of their neighbors
        # <https://stackoverflow.com/a/43096495/353337>
        # num_neighbors = numpy.bincount(mesh.edges['nodes'].flat)
        #
        num_neighbors = numpy.zeros(mesh.node_coords.shape[0], dtype=int)
        new_points = numpy.zeros(mesh.node_coords.shape)
        for edge in mesh.edges['nodes']:
            num_neighbors[edge[0]] += 1
            num_neighbors[edge[1]] += 1
            new_points[edge[0]] += mesh.node_coords[edge[1]]
            new_points[edge[1]] += mesh.node_coords[edge[0]]
        new_points = (new_points.T / num_neighbors).T

        # Keep the boundary vertices in place
        new_points[boundary_verts] = mesh.node_coords[boundary_verts]

        diff = new_points - mesh.node_coords
        max_move = numpy.sqrt(numpy.max(numpy.einsum('ij,ij->i', diff, diff)))

        mesh = MeshTri(
            new_points,
            mesh.cells['nodes'],
            flat_cell_correction=None
            )

        if verbose:
            print('\nstep: {}'.format(k))
            print('  maximum move: {:.15e}'.format(max_move))
            print_stats([gather_stats(mesh)])

    # Flip one last time.
    mesh, _ = flip_until_delaunay(mesh)

    if verbose:
        print('\nBefore:' + 35*' ' + 'After:')
        print_stats([
            initial_stats,
            gather_stats(mesh),
            ])

    if output_filetype:
        write(mesh, output_filetype, num_steps)

    return mesh.node_coords, mesh.cells['nodes']

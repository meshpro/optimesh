# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy

from voropy.mesh_tri import MeshTri


def flip_until_delaunay(mesh):
    if (mesh.get_ce_ratios() > 0).all():
        return mesh, False

    fcc_type = mesh.fcc_type
    if fcc_type is not None:
        # No flat_cell_correction when flipping.
        mesh = MeshTri(
            mesh.node_coords,
            mesh.cells['nodes'],
            flat_cell_correction=None
            )
    mesh.create_edges()
    needs_flipping = numpy.logical_and(
        numpy.logical_not(mesh.is_boundary_edge_individual),
        mesh.get_ce_ratios_per_edge() < 0.0
        )
    is_flipped = numpy.any(needs_flipping)
    k = 0
    while numpy.any(needs_flipping):
        k += 1
        mesh = flip_edges(mesh, needs_flipping)
        #
        mesh.create_edges()
        needs_flipping = numpy.logical_and(
            numpy.logical_not(mesh.is_boundary_edge_individual),
            mesh.get_ce_ratios_per_edge() < 0.0
            )

    # Translate back to input fcc_type.
    if fcc_type is not None:
        mesh = MeshTri(
            mesh.node_coords,
            mesh.cells['nodes'],
            flat_cell_correction=fcc_type
            )
    return mesh, is_flipped


# def flip_for_six(mesh):
#     '''Ideally, all nodes are connected to six neighbors, forming a nicely
#     homogenous mesh. Sometimes, we can flip edges to increase the "six-ness"
#     of a mesh, e.g., if there is a triangle with one node that has less than
#     six, and two nodes that have more than six neighbors.
#     '''
#     # count the number of neighbors
#     mesh.create_edges()
#     num_neighbors = numpy.zeros(len(mesh.node_coords), dtype=int)
#     e = mesh.edges['nodes']
#     numpy.add.at(num_neighbors, e, numpy.ones(e.shape, dtype=int))
#     # Find edges which connect nodes with an adjacency larger than 6. An edge
#     # flip here won't make it worse, and probably will make it better.
#     nn = num_neighbors[e]
#     is_flip_edge = numpy.sum(nn > 6, axis=1) > 1
#     return flip_edges(mesh, is_flip_edge), numpy.any(is_flip_edge)


def flip_edges(mesh, is_flip_edge):
    '''Creates a new mesh by flipping those interior edges which have a
    negative covolume (i.e., a negative covolume-edge length ratio). The
    resulting mesh is Delaunay.
    '''
    is_flip_edge_per_cell = is_flip_edge[mesh.cells['edges']]

    # can only handle the case where each cell has at most one edge to flip
    count = numpy.sum(is_flip_edge_per_cell, axis=1)
    assert all(count <= 1)

    # new cells
    edge_cells = mesh.compute_edge_cells()
    flip_e = numpy.where(is_flip_edge)[0]
    new_cells = numpy.empty((len(flip_e), 2, 3), dtype=int)
    for k, flip_edge in enumerate(flip_e):
        adj_cells = edge_cells[flip_edge]
        assert len(adj_cells) == 2
        # The local edge ids are opposite of the local vertex with the same
        # id.
        cell0_local_edge_id = numpy.where(
            is_flip_edge_per_cell[adj_cells[0]]
            )[0]
        cell1_local_edge_id = numpy.where(
            is_flip_edge_per_cell[adj_cells[1]]
            )[0]

        #     0
        #     /\
        #    /  \
        #   / 0  \
        # 2/______\3
        #  \      /
        #   \  1 /
        #    \  /
        #     \/
        #      1
        verts = [
            mesh.cells['nodes'][adj_cells[0], cell0_local_edge_id],
            mesh.cells['nodes'][adj_cells[1], cell1_local_edge_id],
            mesh.cells['nodes'][adj_cells[0], (cell0_local_edge_id + 1) % 3],
            mesh.cells['nodes'][adj_cells[0], (cell0_local_edge_id + 2) % 3],
            ]
        new_cells[k, 0] = numpy.array([verts[0][0], verts[1][0], verts[2][0]])
        new_cells[k, 1] = numpy.array([verts[0][0], verts[1][0], verts[3][0]])

    # find cells that can stay
    is_good_cell = numpy.all(
        numpy.logical_not(is_flip_edge_per_cell),
        axis=1
        )

    mesh.cells['nodes'] = numpy.concatenate([
        mesh.cells['nodes'][is_good_cell],
        new_cells[:, 0, :],
        new_cells[:, 1, :]
        ])

    # Create new mesh to make sure that all entities are computed again.
    new_mesh = MeshTri(
        mesh.node_coords,
        mesh.cells['nodes'],
        flat_cell_correction=mesh.fcc_type
        )

    return new_mesh


def gather_stats(mesh):
    # The cosines of the angles are the negative dot products of
    # the normalized edges adjacent to the angle.
    norms = numpy.sqrt(mesh.ei_dot_ei)
    normalized_ei_dot_ej = numpy.array([
        mesh.ei_dot_ej[0] / norms[1] / norms[2],
        mesh.ei_dot_ej[1] / norms[2] / norms[0],
        mesh.ei_dot_ej[2] / norms[0] / norms[1],
        ])
    # pylint: disable=invalid-unary-operand-type
    angles = numpy.arccos(-normalized_ei_dot_ej) / (2*numpy.pi) * 360.0

    hist, bin_edges = numpy.histogram(
        angles,
        bins=numpy.linspace(0.0, 180.0, num=19, endpoint=True)
        )
    return hist, bin_edges


def print_stats(data_list):
    # make sure that all data sets have the same length
    n = len(data_list[0][0])
    for data in data_list:
        assert len(data[0]) == n

    # find largest hist value
    max_val = numpy.max([numpy.max(data[0]) for data in data_list])
    digits_max_val = len(str(max_val))
    fmt = (
        9*' '
        + '{{:3.0f}} < angle < {{:3.0f}}:   {{:{:d}d}}'.format(digits_max_val)
        )

    print('  angles (in degrees):\n')
    for i in range(n):
        for data in data_list:
            hist, bin_edges = data
            print(fmt.format(bin_edges[i], bin_edges[i+1], hist[i]), end='')
        print('\n', end='')
    return


def write(mesh, filetype, k):
    if filetype == 'png':
        from matplotlib import pyplot as plt
        fig = mesh.plot(
            show_coedges=False,
            show_centroids=False,
            show_axes=False
            )
        fig.suptitle('step {}'.format(k), fontsize=20)
        plt.savefig('lloyd{:04d}.png'.format(k))
        plt.close(fig)
        return

    mesh.write('lloyd{:04d}.{}'.format(k, filetype))
    return


def sit_in_plane(X, tol=1.0e-15):
    '''Checks if all points X sit in a plane.
    '''
    orth = numpy.cross(X[1] - X[0], X[2] - X[0])
    orth /= numpy.sqrt(numpy.dot(orth, orth))
    return (abs(numpy.dot(X - X[0], orth)) < tol).all()


def get_boundary_edge_ratio(X, cells):
    '''Gets the ratio of the longest vs. the shortest boundary edge.
    '''
    submesh = MeshTri(X, cells, flat_cell_correction='full')
    submesh.create_edges()
    x = submesh.node_coords[
        submesh.idx_hierarchy[..., submesh.is_boundary_edge]
        ]
    e = x[0] - x[1]
    edge_lengths2 = numpy.einsum('ij, ij->i', e, e)
    return numpy.sqrt(max(edge_lengths2) / min(edge_lengths2))


def extract_submesh_entities(X, cells, cell_in_submesh):
    # Get cells
    submesh_cells = cells[cell_in_submesh]
    # Get the vertices
    submesh_verts, uidx = numpy.unique(submesh_cells, return_inverse=True)
    submesh_X = X[submesh_verts]
    #
    submesh_cells = uidx.reshape(submesh_cells.shape)
    return submesh_X, submesh_cells, submesh_verts

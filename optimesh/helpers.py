# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy

from voropy.mesh_tri import MeshTri


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


def write(mesh, file_prefix, filetype, k):
    if filetype == 'png':
        from matplotlib import pyplot as plt
        fig = mesh.plot(
            show_coedges=False,
            show_centroids=False,
            show_axes=False
            )
        fig.suptitle('step {}'.format(k), fontsize=20)
        plt.savefig('{}{:04d}.png'.format(file_prefix, k))
        plt.close(fig)
        return

    mesh.write('{}{:04d}.{}'.format(file_prefix, k, filetype))
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

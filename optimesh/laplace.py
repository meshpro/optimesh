# -*- coding: utf-8 -*-
#
import numpy
import scipy.sparse

from meshplex import MeshTri

from .helpers import runner


def build_adjacency_matrix(mesh):
    i = mesh.idx_hierarchy
    row_idx = i.flat
    col_idx = numpy.array([i[1], i[0]]).flat
    val = numpy.ones(i.shape, dtype=int).flat

    # Create CSR matrix for efficiency
    n = mesh.node_coords.shape[0]
    matrix = scipy.sparse.coo_matrix((val, (row_idx, col_idx)), shape=(n, n))
    matrix = matrix.tocsr()

    # don't count edges more than once
    matrix.data[:] = 1
    return matrix


def fixed_point(points, cells, *args, **kwargs):
    """Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    """
    # The fastfunc approach is a bit faster, but needs fastfunc. Rather fall back to
    # scipy, it's almost as fast, more commonly installed, and Laplace is strictly worse
    # than CPT anyways.
    # def get_new_points(mesh):
    #     import fastfunc
    #     # move interior points into average of their neighbors
    #     num_neighbors = numpy.zeros(len(mesh.node_coords), dtype=int)
    #     idx = mesh.edges["nodes"]
    #     fastfunc.add.at(num_neighbors, idx, numpy.ones(idx.shape, dtype=int))
    #
    #     new_points = numpy.zeros(mesh.node_coords.shape)
    #     fastfunc.add.at(new_points, idx[:, 0], mesh.node_coords[idx[:, 1]])
    #     fastfunc.add.at(new_points, idx[:, 1], mesh.node_coords[idx[:, 0]])
    #
    #     new_points /= num_neighbors[:, None]
    #     idx = mesh.is_boundary_node
    #     new_points[idx] = mesh.node_coords[idx]
    #     return new_points

    def get_new_points(mesh):
        matrix = build_adjacency_matrix(mesh)
        # compute average
        num_neighbors = matrix * numpy.ones(matrix.shape[1], dtype=int)
        new_points = matrix * mesh.node_coords
        new_points /= num_neighbors[:, None]
        # don't move boundary nodes
        idx = mesh.is_boundary_node
        new_points[idx] = mesh.node_coords[idx]
        return new_points

    mesh = MeshTri(points, cells)
    runner(get_new_points, mesh, *args, **kwargs)
    return mesh.node_coords, mesh.cells["nodes"]

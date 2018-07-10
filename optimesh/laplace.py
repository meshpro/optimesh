# -*- coding: utf-8 -*-
#
from __future__ import print_function

import fastfunc
import numpy
import pyamg
import scipy.sparse

from .helpers import runner


def fixed_point(*args, **kwargs):
    """Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    """

    def get_new_points(mesh):
        # move interior points into average of their neighbors
        num_neighbors = numpy.zeros(len(mesh.node_coords), dtype=int)
        idx = mesh.edges["nodes"]
        fastfunc.add.at(num_neighbors, idx, numpy.ones(idx.shape, dtype=int))

        new_points = numpy.zeros(mesh.node_coords.shape)
        fastfunc.add.at(new_points, idx[:, 0], mesh.node_coords[idx[:, 1]])
        fastfunc.add.at(new_points, idx[:, 1], mesh.node_coords[idx[:, 0]])

        idx = mesh.is_interior_node
        new_points = (new_points[idx].T / num_neighbors[idx]).T
        return new_points

    return runner(get_new_points, *args, **kwargs)


def linear_solve(*args, **kwargs):
    """Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    """

    def get_new_points(mesh, tol=1.0e-10):
        cells = mesh.cells["nodes"].T

        row_idx = []
        col_idx = []
        val = []
        a = numpy.ones(cells.shape[1], dtype=float)
        for i in [[0, 1], [1, 2], [2, 0]]:
            edges = cells[i]
            row_idx += [edges[0], edges[1], edges[0], edges[1]]
            col_idx += [edges[0], edges[1], edges[1], edges[0]]
            val += [+a, +a, -a, -a]

        row_idx = numpy.concatenate(row_idx)
        col_idx = numpy.concatenate(col_idx)
        val = numpy.concatenate(val)

        n = mesh.node_coords.shape[0]

        # Create CSR matrix for efficiency
        matrix = scipy.sparse.coo_matrix((val, (row_idx, col_idx)), shape=(n, n))
        matrix = matrix.tocsr()

        # Apply Dirichlet conditions.
        verts = numpy.where(mesh.is_boundary_node)[0]
        # Set all Dirichlet rows to 0.
        for i in verts:
            matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]] = 0.0
        # Set the diagonal and RHS.
        d = matrix.diagonal()
        d[mesh.is_boundary_node] = 1.0
        matrix.setdiag(d)

        rhs = numpy.zeros((n, 2))
        rhs[mesh.is_boundary_node] = mesh.node_coords[mesh.is_boundary_node]

        # out = scipy.sparse.linalg.spsolve(matrix, rhs)
        ml = pyamg.ruge_stuben_solver(matrix)
        # Keep an eye on multiple rhs-solves in pyamg,
        # <https://github.com/pyamg/pyamg/issues/215>.
        out = numpy.column_stack(
            [ml.solve(rhs[:, 0], tol=tol), ml.solve(rhs[:, 1], tol=tol)]
        )
        return out[mesh.is_interior_node]

    return runner(get_new_points, *args, **kwargs)

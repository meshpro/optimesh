# -*- coding: utf-8 -*-
#
import numpy
import pyamg

import scipy.sparse
import scipy.sparse.linalg

from .helpers import runner


def cpt(*args, **kwargs):
    """The `i`th entry in the CPT-energy gradient is

    \\partial E_i = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) \\int_{tau_j} rho

    where d is the dimension of the simplex, `omega_i` is the star of node `i`, `tau_j`
    a triangle in the star, and `b_j` the barycenter of the respective triangle.

    This method aims to preserve the mesh density and hence assumes `rho = 1/|tau|`.
    Incidentally, this makes the integral in the above formula vanish such that the
    energy gradient is a _linear_ function of the mesh coordinates. The linear operator
    is in fact given by the mesh graph Laplacian. Hence, one only needs to solve `d`
    systems with the graph Laplacian (one for each component).

    After this, one does edge-flipping and repeats the solve.
    """

    dim = 2
    alpha = 2 / (dim + 1) ** 2

    def get_new_points(mesh, tol=1.0e-10):
        # Create matrix in IJV format
        # 2 / (dim+1)**2
        edges = mesh.edges["nodes"].T
        row_idx = numpy.concatenate([edges[0], edges[1], edges[0], edges[1]])
        col_idx = numpy.concatenate([edges[0], edges[1], edges[1], edges[0]])
        a = numpy.full(edges.shape[1], alpha)
        val = numpy.concatenate([+a, +a, -a, -a])

        n = mesh.node_coords.shape[0]

        # Set Dirichlet conditions on the boundary
        matrix = scipy.sparse.coo_matrix((val, (row_idx, col_idx)), shape=(n, n))
        # Transform to CSR format for efficiency
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
        out = numpy.column_stack([
            ml.solve(rhs[:, 0], tol=tol),
            ml.solve(rhs[:, 1], tol=tol),
        ])
        return out[mesh.is_interior_node]

    return runner(get_new_points, *args, **kwargs)

# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy

# import pyamg
# import scipy.sparse
from meshplex import MeshTri

from .helpers import runner


def fixed_point_uniform(*args, **kwargs):
    """Lloyd's algorithm.
    """

    def get_new_points(mesh):
        return mesh.control_volume_centroids[mesh.is_interior_node]

    return runner(get_new_points, *args, **kwargs, flat_cell_correction="boundary")


def jac_uniform(mesh):
    # create Jacobian
    centroids = mesh.control_volume_centroids
    X = mesh.node_coords
    jac = 2 * ((X - centroids).T * mesh.control_volumes).T
    return jac.flatten()


def newton_update(mesh):
    X = mesh.node_coords
    cells = mesh.cells["nodes"]

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    i_boundary = numpy.where(mesh.is_boundary_node)[0]

    # Finite difference Jacobian
    eps = 1.0e-5
    X_orig = mesh.node_coords.copy()
    cols = []
    for kk in range(X.shape[0]):
        for kxy in [0, 1]:
            X = X_orig.copy()
            X[kk, kxy] += eps
            jac_plus = jac_uniform(MeshTri(X, cells))
            #
            X = X_orig.copy()
            X[kk, kxy] -= eps
            jac_minus = jac_uniform(MeshTri(X, cells))
            #
            cols.append((jac_plus - jac_minus) / (2 * eps))
    matrix = numpy.column_stack(cols)

    print(numpy.max(numpy.abs(matrix - matrix.T)))

    # Apply Dirichlet conditions.
    for i in numpy.where(mesh.is_boundary_node)[0]:
        matrix[2 * i + 0] = 0.0
        matrix[2 * i + 1] = 0.0
        matrix[2 * i + 0, 2 * i + 0] = 1.0
        matrix[2 * i + 1, 2 * i + 1] = 1.0

    rhs = -jac_uniform(mesh)
    rhs[2 * i_boundary + 0] = 0.0
    rhs[2 * i_boundary + 1] = 0.0

    out = numpy.linalg.solve(matrix, rhs)
    return out.reshape(-1, 2)


def quasi_newton_uniform2(*args, **kwargs):
    """Relaxation with omega. omega=1 leads to Lloyd's algorithm, omega=2 gives good
    results. Check out

    Xiao Xiao,
    Over-Relaxation Lloyd Method For Computing Centroidal Voronoi Tessellations,
    Master's thesis,
    <https://scholarcommons.sc.edu/etd/295/>.

    Everything above omega=2 can lead to flickering, i.e., rapidly alternating updates
    and bad meshes.
    """

    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        omega = 2.0
        x -= omega / 2 * (jac_uniform(mesh).reshape(-1, 2).T / mesh.control_volumes).T
        return x[mesh.is_interior_node]

    return runner(get_new_points, *args, **kwargs, flat_cell_correction="boundary")


def quasi_newton_update_diagonal_blocks(mesh):
    """Lloyd's algorithm can be though of a diagonal-only Hessian; this method
    incorporates the diagonal blocks, too.
    """
    X = mesh.node_coords

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    # Collect the diagonal blocks.
    diagonal_blocks = numpy.zeros((X.shape[0], 2, 2))
    # First the Lloyd part.
    #
    diagonal_blocks[:, 0, 0] += 2 * mesh.control_volumes
    diagonal_blocks[:, 1, 1] += 2 * mesh.control_volumes

    for edges, ce_ratios, ei_outer_ei in zip(
        mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    ):
        # m3 = -0.5 * (ce_ratios * ei_outer_ei.T).T
        for edge, ce in zip(edges, ce_ratios):
            # The diagonal blocks are always positive definite if the mesh is Delaunay.
            i = edge
            ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
            ei_outer_ei = numpy.outer(ei, ei)
            diagonal_blocks[i[0]] -= 0.5 * ce * ei_outer_ei
            diagonal_blocks[i[1]] -= 0.5 * ce * ei_outer_ei

    rhs = -jac_uniform(mesh).reshape(-1, 2)

    return numpy.linalg.solve(diagonal_blocks, rhs)


def quasi_newton_uniform_blocks(*args, **kwargs):
    def get_new_points(mesh):
        # do one Newton step
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += quasi_newton_update_diagonal_blocks(mesh)
        return x[mesh.is_interior_node]

    return runner(get_new_points, *args, **kwargs, flat_cell_correction="boundary")


# def quasi_newton_update_full(mesh):
#     X = mesh.node_coords
#
#     # TODO remove this assertion and test
#     # flat mesh
#     assert X.shape[1] == 2
#
#     i_boundary = numpy.where(mesh.is_boundary_node)[0]
#
#     # create approximate Hessian
#     row_idx = []
#     col_idx = []
#     vals = []
#     for edges, ce_ratios, ei_outer_ei in zip(
#         mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
#     ):
#         # m3 = -0.5 * (ce_ratios * ei_outer_ei.T).T
#         for edge, ce in zip(edges, ce_ratios):
#             # The diagonal blocks are always positive definite if the mesh is Delaunay.
#             i = edge
#             ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
#             ei_outer_ei = numpy.outer(ei, ei)
#             m = -0.5 * ce * ei_outer_ei
#             # (i0, i0) block
#             row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
#             col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
#             vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
#             # (i1, i1) block
#             row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
#             col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
#             vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
#             # if ce < 0.33:
#             #     continue
#             # # (i0, i1) block
#             # row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
#             # col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
#             # vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
#             # # (i1, i0) block
#             # row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
#             # col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
#             # vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
#
#     # add diagonal
#     for k, control_volume in enumerate(mesh.control_volumes):
#         row_idx += [2 * k, 2 * k + 1]
#         col_idx += [2 * k, 2 * k + 1]
#         vals += [2 * control_volume, 2 * control_volume]
#
#     n = mesh.control_volumes.shape[0]
#     matrix = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(2 * n, 2 * n))
#
#     # print()
#     # print(matrix.toarray()[:, k0].reshape(-1))
#     # print()
#     # print(matrix.toarray()[:, 2 * kk + kxy] - 2 * h55)
#     # exit(1)
#
#     # Transform to CSR format for efficiency
#     matrix = matrix.tocsr()
#
#     # print(numpy.sort(numpy.linalg.eigvals(matrix.toarray()))[:5])
#     # exit(1)
#
#     # Apply Dirichlet conditions.
#     # Set all Dirichlet rows to 0.
#     for i in numpy.where(mesh.is_boundary_node)[0]:
#         matrix.data[matrix.indptr[2 * i + 0] : matrix.indptr[2 * i + 0 + 1]] = 0.0
#         matrix.data[matrix.indptr[2 * i + 1] : matrix.indptr[2 * i + 1 + 1]] = 0.0
#     # Set the diagonal and RHS.
#     d = matrix.diagonal()
#     d[2 * i_boundary + 0] = 1.0
#     d[2 * i_boundary + 1] = 1.0
#     matrix.setdiag(d)
#
#     rhs = -jac_uniform(mesh)
#     rhs[2 * i_boundary + 0] = 0.0
#     rhs[2 * i_boundary + 1] = 0.0
#
#     # print("ok hi")
#     # print(numpy.sort(numpy.linalg.eigvals(matrix.toarray())))
#     # exit(1)
#
#     out = scipy.sparse.linalg.spsolve(matrix, rhs)
#     # ml = pyamg.ruge_stuben_solver(matrix)
#     # out = ml.solve(rhs, tol=1.0e-10)
#
#     return out.reshape(-1, 2)
#
#
# def quasi_newton_uniform_full(*args, **kwargs):
#     def get_new_points(mesh):
#         # do one Newton step
#         # TODO need copy?
#         x = mesh.node_coords.copy()
#         x += quasi_newton_update_full(mesh)
#         return x[mesh.is_interior_node]
#
#     return runner(get_new_points, *args, **kwargs, flat_cell_correction="boundary")

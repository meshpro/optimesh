# -*- coding: utf-8 -*-
#
import numpy
import scipy.sparse

from .ghosted_mesh import GhostedMesh
from .helpers import jac_uniform

from ..helpers import runner


def quasi_newton_uniform_full(points, cells, *args, omega=1.0, **kwargs):
    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += update(mesh, omega)
        return x[mesh.is_interior_node]

    ghosted_mesh = GhostedMesh(points, cells)

    runner(
        get_new_points,
        ghosted_mesh.mesh,
        *args,
        **kwargs,
        straighten_out=lambda mesh: ghosted_mesh.straighten_out(),
        get_stats_mesh=lambda mesh: ghosted_mesh.get_stats_mesh(),
    )

    mesh = ghosted_mesh.get_stats_mesh()
    # mesh = ghosted_mesh.mesh
    return mesh.node_coords, mesh.cells["nodes"]


def update(mesh, omega):
    X = mesh.node_coords

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    i_boundary = numpy.where(mesh.is_boundary_node)[0]

    ei_outer_ei = numpy.einsum(
        "ijk, ijl->ijkl", mesh.half_edge_coords, mesh.half_edge_coords
    )

    # create approximate Hessian
    row_idx = []
    col_idx = []
    vals = []

    M = -0.5 * ei_outer_ei * mesh.ce_ratios[..., None, None]
    M_omega = M * omega
    i = mesh.idx_hierarchy

    for k in range(3):
        # The diagonal blocks are always positive definite if the mesh is Delaunay.
        # (i0, i0) block
        row_idx += [2 * i[0, k] + 0, 2 * i[0, k] + 0, 2 * i[0, k] + 1, 2 * i[0, k] + 1]
        col_idx += [2 * i[0, k] + 0, 2 * i[0, k] + 1, 2 * i[0, k] + 0, 2 * i[0, k] + 1]
        vals += [M[k, :, 0, 0], M[k, :, 0, 1], M[k, :, 1, 0], M[k, :, 1, 1]]
        # (i1, i1) block
        row_idx += [2 * i[1, k] + 0, 2 * i[1, k] + 0, 2 * i[1, k] + 1, 2 * i[1, k] + 1]
        col_idx += [2 * i[1, k] + 0, 2 * i[1, k] + 1, 2 * i[1, k] + 0, 2 * i[1, k] + 1]
        vals += [M[k, :, 0, 0], M[k, :, 0, 1], M[k, :, 1, 0], M[k, :, 1, 1]]
        # Scale the off-diagonal blocks with some factor. If omega == 1, this is the
        # Hessian. Unfortunately, it seems that Newton domain of convergence is
        # really small. The relaxation makes the method more robust.
        # (i0, i1) block
        row_idx += [2 * i[0, k] + 0, 2 * i[0, k] + 0, 2 * i[0, k] + 1, 2 * i[0, k] + 1]
        col_idx += [2 * i[1, k] + 0, 2 * i[1, k] + 1, 2 * i[1, k] + 0, 2 * i[1, k] + 1]
        vals += [M_omega[k, :, 0, 0], M_omega[k, :, 0, 1], M_omega[k, :, 1, 0], M_omega[k, :, 1, 1]]
        # (i1, i0) block
        row_idx += [2 * i[1, k] + 0, 2 * i[1, k] + 0, 2 * i[1, k] + 1, 2 * i[1, k] + 1]
        col_idx += [2 * i[0, k] + 0, 2 * i[0, k] + 1, 2 * i[0, k] + 0, 2 * i[0, k] + 1]
        vals += [M_omega[k, :, 0, 0], M_omega[k, :, 0, 1], M_omega[k, :, 1, 0], M_omega[k, :, 1, 1]]

    # add diagonal
    n = mesh.control_volumes.shape[0]
    row_idx.append(2 * numpy.arange(n))
    col_idx.append(2 * numpy.arange(n))
    vals.append(2 * mesh.control_volumes)
    #
    row_idx.append(2 * numpy.arange(n) + 1)
    col_idx.append(2 * numpy.arange(n) + 1)
    vals.append(2 * mesh.control_volumes)

    row_idx = numpy.concatenate(row_idx)
    col_idx = numpy.concatenate(col_idx)
    vals = numpy.concatenate(vals)

    matrix = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(2 * n, 2 * n))

    # Transform to CSR format for efficiency
    matrix = matrix.tocsr()

    # Apply Dirichlet conditions.
    # Set all Dirichlet rows to 0.
    for i in numpy.where(mesh.is_boundary_node)[0]:
        matrix.data[matrix.indptr[2 * i + 0] : matrix.indptr[2 * i + 0 + 1]] = 0.0
        matrix.data[matrix.indptr[2 * i + 1] : matrix.indptr[2 * i + 1 + 1]] = 0.0
    # Set the diagonal and RHS.
    d = matrix.diagonal()
    d[2 * i_boundary + 0] = 1.0
    d[2 * i_boundary + 1] = 1.0
    matrix.setdiag(d)

    rhs = -jac_uniform(mesh)
    rhs[2 * i_boundary + 0] = 0.0
    rhs[2 * i_boundary + 1] = 0.0

    # print("ok hi")
    # print(numpy.sort(numpy.linalg.eigvals(matrix.toarray())))
    # exit(1)

    out = scipy.sparse.linalg.spsolve(matrix, rhs)
    # ml = pyamg.ruge_stuben_solver(matrix)
    # out = ml.solve(rhs, tol=1.0e-10)

    return out.reshape(-1, 2)

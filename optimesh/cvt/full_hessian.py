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
        return x

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
    ei_outer_ei = numpy.einsum(
        "ijk, ijl->ijkl", mesh.half_edge_coords, mesh.half_edge_coords
    )

    # create approximate Hessian
    row_idx = []
    col_idx = []
    vals = []

    M = -0.5 * ei_outer_ei * mesh.ce_ratios[..., None, None]
    # Scale the off-diagonal blocks with some factor. If omega == 1, this is the
    # Hessian. Unfortunately, it seems that the Newton domain of convergence is small.
    # Relaxation makes the method more robust.
    M2 = M * omega

    block_size = M.shape[2]
    assert block_size == M.shape[3]

    for k in range(M.shape[0]):
        # The diagonal blocks are always positive definite if the mesh is Delaunay.
        # (i0, i0) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[0, k] + i]
                col_idx += [block_size * mesh.idx_hierarchy[0, k] + j]
                vals += [M[k, :, i, j]]
        # (i1, i1) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[1, k] + i]
                col_idx += [block_size * mesh.idx_hierarchy[1, k] + j]
                vals += [M[k, :, i, j]]
        # (i0, i1) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[0, k] + i]
                col_idx += [block_size * mesh.idx_hierarchy[1, k] + j]
                vals += [M2[k, :, i, j]]
        # (i1, i0) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[1, k] + i]
                col_idx += [block_size * mesh.idx_hierarchy[0, k] + j]
                vals += [M2[k, :, i, j]]

    # add diagonal
    n = mesh.control_volumes.shape[0]
    for k in range(block_size):
        row_idx.append(block_size * numpy.arange(n) + k)
        col_idx.append(block_size * numpy.arange(n) + k)
        vals.append(2 * mesh.control_volumes)

    row_idx = numpy.concatenate(row_idx)
    col_idx = numpy.concatenate(col_idx)
    vals = numpy.concatenate(vals)

    matrix = scipy.sparse.coo_matrix(
        (vals, (row_idx, col_idx)), shape=(block_size * n, block_size * n)
    )

    # Transform to CSR format for efficiency
    matrix = matrix.tocsr()

    # Apply Dirichlet conditions.
    # Set all Dirichlet rows to 0.
    i_boundary = numpy.where(mesh.is_boundary_node)[0]
    for i in i_boundary:
        for k in range(block_size):
            s = block_size * i + k
            matrix.data[matrix.indptr[s] : matrix.indptr[s + 1]] = 0.0
    # Set the diagonal and RHS.
    d = matrix.diagonal()
    for k in range(block_size):
        d[block_size * i_boundary + k] = 1.0
    matrix.setdiag(d)
    #
    rhs = -jac_uniform(mesh)
    rhs[i_boundary] = 0.0
    rhs = rhs.reshape(-1)

    out = scipy.sparse.linalg.spsolve(matrix, rhs)
    # import pyamg
    # ml = pyamg.ruge_stuben_solver(matrix)
    # out = ml.solve(rhs, tol=1.0e-12)

    return out.reshape(-1, block_size)

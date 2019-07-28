# -*- coding: utf-8 -*-
#
import numpy
import scipy.sparse

from meshplex import MeshTri

from ..helpers import runner
from .helpers import jac_uniform


def quasi_newton_uniform_full(points, cells, *args, **kwargs):
    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += update(mesh)
        return x

    mesh = MeshTri(points, cells)

    out = runner(
        get_new_points,
        mesh,
        *args,
        **kwargs,
        method_name="Centroidal Voronoi Tesselation (CVT), uniform density, exact-Hessian variant"
    )
    print(out)
    return mesh.node_coords, mesh.cells["nodes"]


def update(mesh):
    # Exclude all cells which have a too negative covolume-edgelength ratio. This is
    # necessary to prevent nodes to be dragged outside of the domain by very flat
    # cells on the boundary.
    # There are other possible heuristics too. For example, one could restrict the
    # mask to cells at or near the boundary.
    # TODO It seems that we would need other criteria to make it even more robust
    mask = numpy.any(mesh.ce_ratios < -0.5, axis=0)
    # mask = numpy.zeros(mesh.ce_ratios.shape[1], dtype=bool)

    hec = mesh.half_edge_coords[:, ~mask]
    ei_outer_ei = numpy.einsum("...k,...l->...kl", hec, hec)

    # create approximate Hessian
    row_idx = []
    col_idx = []
    vals = []

    M = -0.5 * ei_outer_ei * mesh.ce_ratios[:, ~mask, None, None]

    block_size = M.shape[2]
    assert block_size == M.shape[3]

    for k in range(M.shape[0]):
        # The diagonal blocks are always positive definite if the mesh is Delaunay.
        # (i0, i0) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[0, k, ~mask] + i]
                col_idx += [block_size * mesh.idx_hierarchy[0, k, ~mask] + j]
                vals += [M[k, :, i, j]]
        # (i1, i1) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[1, k, ~mask] + i]
                col_idx += [block_size * mesh.idx_hierarchy[1, k, ~mask] + j]
                vals += [M[k, :, i, j]]
        # (i0, i1) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[0, k, ~mask] + i]
                col_idx += [block_size * mesh.idx_hierarchy[1, k, ~mask] + j]
                vals += [M[k, :, i, j]]
        # (i1, i0) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * mesh.idx_hierarchy[1, k, ~mask] + i]
                col_idx += [block_size * mesh.idx_hierarchy[0, k, ~mask] + j]
                vals += [M[k, :, i, j]]

    # add diagonal
    cv = mesh.get_control_volumes(cell_mask=mask)
    n = cv.shape[0]
    for k in range(block_size):
        row_idx.append(block_size * numpy.arange(n) + k)
        col_idx.append(block_size * numpy.arange(n) + k)
        vals.append(2 * cv)

    row_idx = numpy.concatenate(row_idx)
    col_idx = numpy.concatenate(col_idx)
    vals = numpy.concatenate(vals)

    matrix = scipy.sparse.coo_matrix(
        (vals, (row_idx, col_idx)), shape=(block_size * n, block_size * n)
    )

    # Transform to CSR format for efficiency
    matrix = matrix.tocsr()
    #
    rhs = -jac_uniform(mesh, mask)

    # Apply Dirichlet conditions.
    # Set all Dirichlet rows to 0. When using a cell mask, it can happen that some nodes
    # don't get any contribution at all because they are adjacent only to masked cells.
    # Reset those, too.
    idx = numpy.any(numpy.isnan(rhs), axis=1) | mesh.is_boundary_node
    i_reset = numpy.where(idx)[0]
    for i in i_reset:
        for k in range(block_size):
            s = block_size * i + k
            matrix.data[matrix.indptr[s] : matrix.indptr[s + 1]] = 0.0
    # Set the diagonal and RHS.
    d = matrix.diagonal()
    for k in range(block_size):
        d[block_size * i_reset + k] = 1.0
    matrix.setdiag(d)
    rhs[i_reset] = 0.0
    rhs = rhs.reshape(-1)

    out = scipy.sparse.linalg.spsolve(matrix, rhs)
    # import pyamg
    # ml = pyamg.ruge_stuben_solver(matrix)
    # out = ml.solve(rhs, tol=1.0e-12)

    return out.reshape(-1, block_size)

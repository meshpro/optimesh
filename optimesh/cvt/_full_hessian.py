import numpy
import scipy.sparse

from meshplex import MeshTri

from ..helpers import runner
from ._helpers import jac_uniform


def quasi_newton_uniform_full(points, cells, *args, **kwargs):
    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += update(mesh)
        return x

    mesh = MeshTri(points, cells)

    runner(
        get_new_points,
        mesh,
        *args,
        **kwargs,
        method_name="Centroidal Voronoi Tesselation (CVT), uniform density, full-Hessian variant"
    )
    return mesh.node_coords, mesh.cells["nodes"]


def update(mesh):
    # Exclude all cells which have a too negative covolume-edgelength ratio. This is
    # necessary to prevent nodes to be dragged outside of the domain by very flat
    # cells on the boundary.
    # There are other possible heuristics too. For example, one could restrict the
    # mask to cells at or near the boundary.
    mask = numpy.any(mesh.ce_ratios < -0.5, axis=0)

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
        idx0 = mesh.idx_hierarchy[0, k, ~mask]
        idx1 = mesh.idx_hierarchy[1, k, ~mask]
        # The diagonal blocks are always positive definite if the mesh is Delaunay.
        # (i0, i0) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * idx0 + i]
                col_idx += [block_size * idx0 + j]
                vals += [M[k, :, i, j]]
        # (i1, i1) block
        for i in range(block_size):
            for j in range(block_size):
                row_idx += [block_size * idx1 + i]
                col_idx += [block_size * idx1 + j]
                vals += [M[k, :, i, j]]

        # This is a cheap workaround.
        # It turns out that this method still isn't very robust against flat cells near
        # the boundary. The related Lloyd method and the block-diagonal method, however,
        # are. Hence, if there is any masked cell, use the block variant for robustness.
        # (This corresponds to eliminating all off-diagonal blocks.)
        # TODO find a better criterion
        if ~numpy.any(mask):
            # (i0, i1) block
            for i in range(block_size):
                for j in range(block_size):
                    row_idx += [block_size * idx0 + i]
                    col_idx += [block_size * idx1 + j]
                    vals += [M[k, :, i, j]]
            # (i1, i0) block
            for i in range(block_size):
                for j in range(block_size):
                    row_idx += [block_size * idx1 + i]
                    col_idx += [block_size * idx0 + j]
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
    # Ideally, we'd allow the boundary nodes to move, too, and move them back to the
    # boundary in a second step. Since the points are coupled, however, the interior
    # points would move to different places as well.
    #
    # Instead of hard Dirichlet conditions, we would have to insert conditions for the
    # points to move along the surface. Before this is implemented, just use Dirichlet.
    #
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

    out = scipy.sparse.linalg.spsolve(matrix, rhs.reshape(-1))
    # import pyamg
    # ml = pyamg.ruge_stuben_solver(matrix)
    # out = ml.solve(rhs, tol=1.0e-12)
    dX = out.reshape(-1, block_size)

    # idx = numpy.any(numpy.isnan(rhs), axis=1) | mesh.is_boundary_node
    # dX[idx] = 0.0

    return dX

# -*- coding: utf-8 -*-
#
from __future__ import print_function

import copy

import numpy

# import pyamg
# import scipy.sparse
from meshplex import MeshTri

from .helpers import runner, print_stats


def _row_dot(a, b):
    # https://stackoverflow.com/a/26168677/353337
    return numpy.einsum("ij, ij->i", a, b)


def _reflect_point(p0, p1, p2):
    """For any given triangle p0--p1--p2, this method creates the point p0',
    namely p0 reflected along the edge p1--p2, and the point q at the
    perpendicular intersection of the reflection.

            p0
          _/| \__
        _/  |    \__
       /    |       \
      p1----|q-------p2
       \_   |     __/
         \_ |  __/
           \| /
           p0'

    """
    if len(p0) == 0:
        return numpy.empty(p0.shape), numpy.empty(p0.shape)
    # TODO cache some of the entities here (the ones related to p1 and p2
    alpha = _row_dot(p0 - p1, p2 - p1) / _row_dot(p2 - p1, p2 - p1)
    # q: Intersection point of old and new edge
    # q = p1 + dot(p0-p1, (p2-p1)/||p2-p1||) * (p2-p1)/||p2-p1||
    #   = p1 + dot(p0-p1, p2-p1)/dot(p2-p1, p2-p1) * (p2-p1)
    q = p1 + alpha[:, None] * (p2 - p1)
    # p0' = p0 + 2*(q - p0)
    return 2 * q - p0


def fixed_point_uniform(points, cells, *args, **kwargs):
    """Lloyd's algorithm.
    """
    assert points.shape[1] == 2

    def reflect_ghost_points(mesh):
        # The ghost mirror facet points ghost_mirror[1], ghost_mirror[2] are on the
        # boundary and never change in any way. The point that is mirrored,
        # ghost_mirror[0], however, does move and may after a facet flip even refer to
        # an entirely different point. Find out which.
        # mirrors = numpy.empty(num_boundary_cells,_dtype=int)
        mirrors = numpy.zeros(num_boundary_cells, dtype=int)

        new_edges_nodes = mesh.edges["nodes"][ghost_edge_gids]
        has_flipped = original_edges_nodes[:, 0] != new_edges_nodes[:, 0]

        # In the beginning, the ghost points are appended to the points array and hence
        # have higher GIDs than all other points. Since the edges["nodes"] are sorted,
        # the second entry must be the ghost point.
        assert numpy.all(is_ghost_point[new_edges_nodes[has_flipped, 1]]), \
            "A ghost edge is flipped, but does not contain the ghost point. This " \
            "usually indicates that the initial mesh is *very* weird around the " \
            "boundary. Try applying one or two CPT steps first."
        # The first point is the one at the other end of the flipped edge.
        mirrors[has_flipped] = new_edges_nodes[has_flipped, 0]

        # Now let's look at the ghost points whose edge has _not_ flipped. We need to
        # find the cell on the other side and there the point opposite of the ghost
        # edge.
        num_adjacent_cells, interior_edge_idx = mesh.edge_gid_to_edge_list[
            ghost_edge_gids
        ][~has_flipped].T
        assert numpy.all(num_adjacent_cells == 2)
        #
        adj_cells = mesh.edges_cells[2][interior_edge_idx]
        is_first = adj_cells[:, 0] == ghost_cell_gids[~has_flipped]
        #
        is_second = adj_cells[:, 1] == ghost_cell_gids[~has_flipped]
        assert numpy.all(numpy.logical_xor(is_first, is_second))
        #
        opposite_cell_id = numpy.empty(adj_cells.shape[0], dtype=int)
        opposite_cell_id[is_first] = adj_cells[is_first, 1]
        opposite_cell_id[is_second] = adj_cells[is_second, 0]
        # Now find the cell opposite of the ghost edge in the oppisite cell.
        eq = numpy.array(
            [
                mesh.cells["edges"][opposite_cell_id, k]
                == ghost_edge_gids[~has_flipped]
                for k in range(mesh.cells["edges"].shape[1])
            ]
        )
        assert numpy.all(numpy.sum(eq, axis=0) == 1)
        opposite_node_id = numpy.empty(eq.shape[1], dtype=int)
        cn = mesh.cells["nodes"][opposite_cell_id]
        for k in range(eq.shape[0]):
            opposite_node_id[eq[k]] = cn[eq[k], k]
        # Set in mirrors
        mirrors[~has_flipped] = opposite_node_id

        # finally get the reflection
        pts = mesh.node_coords
        mp = _reflect_point(pts[mirrors], pts[ghost_mirror[1]], pts[ghost_mirror[2]])
        return mp

    def get_new_points(mesh):
        return mesh.control_volume_centroids[mesh.is_interior_node]

    # Add ghost points and cells for boundary facets
    msh = MeshTri(points, cells)
    ghost_mirror = []
    ghost_cells = []
    k = points.shape[0]
    for i in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        bf = msh.is_boundary_facet[i[0]]
        c = msh.cells["nodes"][bf].T
        ghost_mirror.append(c[i])
        n = c.shape[1]
        p = numpy.arange(k, k + n)
        ghost_cells.append(numpy.column_stack([p, c[i[1]], c[i[2]]]))
        k += n

    num_boundary_cells = numpy.sum(msh.is_boundary_facet)

    is_ghost_point = numpy.zeros(points.shape[0] + num_boundary_cells, dtype=bool)
    ghost_point_gids = numpy.arange(
        points.shape[0], points.shape[0] + num_boundary_cells
    )
    is_ghost_point[points.shape[0] :] = True
    is_ghost_cell = numpy.zeros(cells.shape[0] + num_boundary_cells, dtype=bool)
    ghost_cell_gids = numpy.arange(cells.shape[0], cells.shape[0] + num_boundary_cells)
    is_ghost_cell[cells.shape[0] :] = True

    ghost_mirror = numpy.concatenate(ghost_mirror, axis=1)
    assert ghost_mirror.shape[1] == num_boundary_cells

    num_original_points = points.shape[0]
    points = numpy.concatenate(
        [points, numpy.zeros((num_boundary_cells, points.shape[1]))]
    )
    num_original_cells = cells.shape[0]
    cells = numpy.concatenate([cells, *ghost_cells])

    # Set ghost points
    points[is_ghost_point] = _reflect_point(
        points[ghost_mirror[0]], points[ghost_mirror[1]], points[ghost_mirror[2]]
    )

    # Create new mesh, remember the pseudo-boundary edges
    mesh = MeshTri(points, cells)

    mesh.create_edges()
    # Get the first edge in the ghost cells. (The first point is the ghost point, and
    # opposite edges have the same index.)
    ghost_edge_gids = mesh.cells["edges"][is_ghost_cell, 0]
    original_edges_nodes = mesh.edges["nodes"][ghost_edge_gids]

    def get_flip_ghost_edges(mesh):
        # unflip boundary edges
        new_edges_nodes = mesh.edges["nodes"][ghost_edge_gids]
        # Either both nodes are equal or both are unequal; it's enough to check just the
        # first column.
        flip_edge_gids = ghost_edge_gids[
            original_edges_nodes[:, 0] != new_edges_nodes[:, 0]
        ]
        # Assert that this is actually an interior edge in the ghosted mesh, and get the
        # index into the interior edge array of the mesh.
        num_adjacent_cells, flip_interior_edge_idx = mesh.edge_gid_to_edge_list[
            flip_edge_gids
        ].T
        assert numpy.all(num_adjacent_cells == 2)
        return flip_interior_edge_idx

    # def get_stats_mesh(mesh):
    #     # Make deep copy to avoid influencing the actual mesh
    #     print("stats_mesh ghost points")
    #     print(mesh.node_coords[is_ghost_point])
    #     return mesh

    def straighten_out(mesh):
        mesh.flip_until_delaunay()
        mesh.node_coords[is_ghost_point] = reflect_ghost_points(mesh)
        mesh.update_values()
        return

    runner(
        get_new_points,
        mesh,
        *args,
        **kwargs,
        straighten_out=straighten_out,
    )

    return mesh.node_coords, mesh.cells["nodes"]


def jac_uniform(mesh):
    # create Jacobian
    centroids = mesh.control_volume_centroids
    X = mesh.node_coords
    jac = 2 * ((X - centroids).T * mesh.control_volumes).T
    return jac.flatten()


# def newton_update(mesh):
#     X = mesh.node_coords
#     cells = mesh.cells["nodes"]
#
#     # TODO remove this assertion and test
#     # flat mesh
#     assert X.shape[1] == 2
#
#     i_boundary = numpy.where(mesh.is_boundary_node)[0]
#
#     # Finite difference Jacobian
#     eps = 1.0e-5
#     X_orig = mesh.node_coords.copy()
#     cols = []
#     for kk in range(X.shape[0]):
#         for kxy in [0, 1]:
#             X = X_orig.copy()
#             X[kk, kxy] += eps
#             jac_plus = jac_uniform(MeshTri(X, cells))
#             #
#             X = X_orig.copy()
#             X[kk, kxy] -= eps
#             jac_minus = jac_uniform(MeshTri(X, cells))
#             #
#             cols.append((jac_plus - jac_minus) / (2 * eps))
#     matrix = numpy.column_stack(cols)
#
#     print(numpy.max(numpy.abs(matrix - matrix.T)))
#
#     # Apply Dirichlet conditions.
#     for i in numpy.where(mesh.is_boundary_node)[0]:
#         matrix[2 * i + 0] = 0.0
#         matrix[2 * i + 1] = 0.0
#         matrix[2 * i + 0, 2 * i + 0] = 1.0
#         matrix[2 * i + 1, 2 * i + 1] = 1.0
#
#     rhs = -jac_uniform(mesh)
#     rhs[2 * i_boundary + 0] = 0.0
#     rhs[2 * i_boundary + 1] = 0.0
#
#     out = numpy.linalg.solve(matrix, rhs)
#     return out.reshape(-1, 2)


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

    return runner(get_new_points, *args, **kwargs)


def quasi_newton_update_diagonal_blocks(mesh):
    """Lloyd's algorithm can be though of a diagonal-only Hessian; this method
    incorporates the diagonal blocks, too.
    """
    X = mesh.node_coords

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    assert numpy.all(mesh.ce_ratios_per_interior_edge > 0)

    # Collect the diagonal blocks.
    diagonal_blocks = numpy.zeros((X.shape[0], 2, 2))

    # print(mesh.control_volumes)
    # First the Lloyd part.
    diagonal_blocks[:, 0, 0] += 2 * mesh.control_volumes
    diagonal_blocks[:, 1, 1] += 2 * mesh.control_volumes
    # print("diagonal_blocks_orig")
    # print(diagonal_blocks_orig)

    #  diagonal_blocks = numpy.zeros((X.shape[0], 2, 2))
    # for edges, ce_ratios, ei_outer_ei in zip(
    #     mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    # ):
    #     for i, ce in zip(edges, ce_ratios):
    #         ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
    #         m = numpy.eye(2) * 0.5 * ce * numpy.dot(ei, ei)
    #         diagonal_blocks[i[0]] += m
    #         diagonal_blocks[i[1]] += m

    # print()
    # print("diagonal_blocks")
    # print(diagonal_blocks)
    # # exit(1)
    # print()
    # print("difference")
    # diff = diagonal_blocks - diagonal_blocks_orig
    # print(diff)
    # print()

    # assert numpy.all(numpy.abs(diff) < 1.0e-10)

    for edges, ce_ratios, ei_outer_ei in zip(
        mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    ):
        # m3 = -0.5 * (ce_ratios * ei_outer_ei.T).T
        for i, ce in zip(edges, ce_ratios):
            # The diagonal blocks are always positive definite if the mesh is Delaunay.
            ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
            m = -0.5 * ce * numpy.outer(ei, ei)
            diagonal_blocks[i[0]] += m
            diagonal_blocks[i[1]] += m
            #
            m = 0.5 * ce * (numpy.eye(2) * numpy.dot(ei, ei) - numpy.outer(ei, ei))
            assert numpy.all(ce * numpy.linalg.eigvalsh(m) > -1.0e-12)
            # diagonal_blocks[i[0]] += m
            # diagonal_blocks[i[1]] += m

    rhs = -jac_uniform(mesh).reshape(-1, 2)

    for ibn, block in zip(mesh.is_boundary_node, diagonal_blocks):
        eigvals = numpy.linalg.eigvalsh(block)
        assert numpy.all(eigvals > 0.0), eigvals
    # print()

    return numpy.linalg.solve(diagonal_blocks, rhs)


def quasi_newton_uniform_blocks(*args, **kwargs):
    def get_new_points(mesh):
        # do one Newton step
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += quasi_newton_update_diagonal_blocks(mesh)
        return x[mesh.is_interior_node]

    return runner(get_new_points, *args, **kwargs)


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
#             omega = 0.90
#             m = -0.5 * ce * ei_outer_ei * omega
#             # (i0, i0) block
#             row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
#             col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
#             vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
#             # (i1, i1) block
#             row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
#             col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
#             vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
#             # (i0, i1) block
#             row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
#             col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
#             vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
#             # (i1, i0) block
#             row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
#             col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
#             vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
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
#     return runner(get_new_points, *args, **kwargs)

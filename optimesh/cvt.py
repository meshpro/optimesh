# -*- coding: utf-8 -*-
#
from __future__ import print_function

import copy

import numpy

# import pyamg
import scipy.sparse
from meshplex import MeshTri

from .helpers import runner


class GhostedMesh(object):
    def __init__(self, points, cells):
        # Add ghost points and cells for boundary facets
        msh = MeshTri(points, cells)
        self.ghost_mirror = []
        ghost_cells = []
        k = points.shape[0]
        for i in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
            bf = msh.is_boundary_facet[i[0]]
            c = msh.cells["nodes"][bf].T
            self.ghost_mirror.append(c[i])
            n = c.shape[1]
            p = numpy.arange(k, k + n)
            ghost_cells.append(numpy.column_stack([p, c[i[1]], c[i[2]]]))
            k += n

        self.num_boundary_cells = numpy.sum(msh.is_boundary_facet)

        self.is_ghost_point = numpy.zeros(
            points.shape[0] + self.num_boundary_cells, dtype=bool
        )
        self.is_ghost_point[points.shape[0] :] = True
        is_ghost_cell = numpy.zeros(
            cells.shape[0] + self.num_boundary_cells, dtype=bool
        )
        self.ghost_cell_gids = numpy.arange(
            cells.shape[0], cells.shape[0] + self.num_boundary_cells
        )
        is_ghost_cell[cells.shape[0] :] = True

        self.ghost_mirror = numpy.concatenate(self.ghost_mirror, axis=1)
        assert self.ghost_mirror.shape[1] == self.num_boundary_cells

        self.num_original_points = points.shape[0]
        points = numpy.concatenate(
            [points, numpy.zeros((self.num_boundary_cells, points.shape[1]))]
        )
        self.num_original_cells = cells.shape[0]
        cells = numpy.concatenate([cells, *ghost_cells])

        # Set ghost points
        points[self.is_ghost_point] = _reflect_point(
            points[self.ghost_mirror[0]],
            points[self.ghost_mirror[1]],
            points[self.ghost_mirror[2]],
        )

        # Create new mesh, remember the pseudo-boundary edges
        self.mesh = MeshTri(points, cells)

        self.mesh.create_edges()
        # Get the first edge in the ghost cells. (The first point is the ghost point,
        # and opposite edges have the same index.)
        self.ghost_edge_gids = self.mesh.cells["edges"][is_ghost_cell, 0]
        self.original_edges_nodes = self.mesh.edges["nodes"][self.ghost_edge_gids]
        return

    def get_flip_ghost_edges(self):
        # unflip boundary edges
        new_edges_nodes = self.mesh.edges["nodes"][self.ghost_edge_gids]
        # Either both nodes are equal or both are unequal; it's enough to check just the
        # first column.
        flip_edge_gids = self.ghost_edge_gids[
            self.original_edges_nodes[:, 0] != new_edges_nodes[:, 0]
        ]
        # Assert that this is actually an interior edge in the ghosted mesh, and get the
        # index into the interior edge array of the mesh.
        num_adjacent_cells, flip_interior_edge_idx = self.mesh.edge_gid_to_edge_list[
            flip_edge_gids
        ].T
        assert numpy.all(num_adjacent_cells == 2)
        return flip_interior_edge_idx

    def reflect_ghost_points(self):
        # The ghost mirror facet points ghost_mirror[1], ghost_mirror[2] are on the
        # boundary and never change in any way. The point that is mirrored,
        # ghost_mirror[0], however, does move and may after a facet flip even refer to
        # an entirely different point. Find out which.
        # mirrors = numpy.empty(num_boundary_cells,_dtype=int)
        mirrors = numpy.zeros(self.num_boundary_cells, dtype=int)

        new_edges_nodes = self.mesh.edges["nodes"][self.ghost_edge_gids]
        has_flipped = self.original_edges_nodes[:, 0] != new_edges_nodes[:, 0]

        # In the beginning, the ghost points are appended to the points array and hence
        # have higher GIDs than all other points. Since the edges["nodes"] are sorted,
        # the second entry must be the ghost point.
        assert numpy.all(self.is_ghost_point[new_edges_nodes[has_flipped, 1]]), (
            "A ghost edge is flipped, but does not contain the ghost point. Try "
            "applying some steps of a more robust method first."
        )
        # The first point is the one at the other end of the flipped edge.
        mirrors[has_flipped] = new_edges_nodes[has_flipped, 0]

        # Now let's look at the ghost points whose edge has _not_ flipped. We need to
        # find the cell on the other side and there the point opposite of the ghost
        # edge.
        num_adjacent_cells, interior_edge_idx = self.mesh.edge_gid_to_edge_list[
            self.ghost_edge_gids
        ][~has_flipped].T
        assert numpy.all(num_adjacent_cells == 2)
        #
        adj_cells = self.mesh.edges_cells[2][interior_edge_idx]
        is_first = adj_cells[:, 0] == self.ghost_cell_gids[~has_flipped]
        #
        is_second = adj_cells[:, 1] == self.ghost_cell_gids[~has_flipped]
        assert numpy.all(numpy.logical_xor(is_first, is_second))
        #
        opposite_cell_id = numpy.empty(adj_cells.shape[0], dtype=int)
        opposite_cell_id[is_first] = adj_cells[is_first, 1]
        opposite_cell_id[is_second] = adj_cells[is_second, 0]
        # Now find the cell opposite of the ghost edge in the oppisite cell.
        eq = numpy.array(
            [
                self.mesh.cells["edges"][opposite_cell_id, k]
                == self.ghost_edge_gids[~has_flipped]
                for k in range(self.mesh.cells["edges"].shape[1])
            ]
        )
        assert numpy.all(numpy.sum(eq, axis=0) == 1)
        opposite_node_id = numpy.empty(eq.shape[1], dtype=int)
        cn = self.mesh.cells["nodes"][opposite_cell_id]
        for k in range(eq.shape[0]):
            opposite_node_id[eq[k]] = cn[eq[k], k]
        # Set in mirrors
        mirrors[~has_flipped] = opposite_node_id

        # finally get the reflection
        pts = self.mesh.node_coords
        mp = _reflect_point(
            pts[mirrors], pts[self.ghost_mirror[1]], pts[self.ghost_mirror[2]]
        )
        return mp

    def get_stats_mesh(self):
        # Make deep copy to avoid influencing the actual mesh
        mesh2 = copy.deepcopy(self.mesh)
        mesh2.flip_interior_edges(self.get_flip_ghost_edges())
        # remove ghost cells
        # TODO this is too crude; sometimes the wrong cells are cut
        points = mesh2.node_coords[: self.num_original_points]
        cells = mesh2.cells["nodes"][: self.num_original_cells]
        return MeshTri(points, cells)

    def straighten_out(self):
        self.mesh.flip_until_delaunay()
        self.mesh.node_coords[self.is_ghost_point] = self.reflect_ghost_points()
        self.mesh.update_values()
        return


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


# def fixed_point_uniform(points, cells, *args, **kwargs):
#     """Lloyd's algorithm.
#     """
#     assert points.shape[1] == 2
#
#     def get_new_points(mesh):
#         return mesh.control_volume_centroids[mesh.is_interior_node]
#
#     ghosted_mesh = GhostedMesh(points, cells)
#
#     runner(
#         get_new_points,
#         ghosted_mesh.mesh,
#         *args,
#         **kwargs,
#         straighten_out=lambda mesh: ghosted_mesh.straighten_out(),
#         # get_stats_mesh=lambda mesh: ghosted_mesh.get_stats_mesh(),
#     )
#
#     mesh = ghosted_mesh.get_stats_mesh()
#     return mesh.node_coords, mesh.cells["nodes"]


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


def quasi_newton_uniform_lloyd(points, cells, *args, omega=1.0, **kwargs):
    """Relaxed Lloyd's algorithm. omega=1 leads to Lloyd's algorithm, overrelaxation
    omega=2 gives good results. Check out

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
        x -= omega / 2 * (jac_uniform(mesh).reshape(-1, 2).T / mesh.control_volumes).T
        return x[mesh.is_interior_node]

    ghosted_mesh = GhostedMesh(points, cells)

    runner(
        get_new_points,
        ghosted_mesh.mesh,
        *args,
        **kwargs,
        straighten_out=lambda mesh: ghosted_mesh.straighten_out(),
        # get_stats_mesh=lambda mesh: ghosted_mesh.get_stats_mesh(),
    )

    mesh = ghosted_mesh.get_stats_mesh()
    return mesh.node_coords, mesh.cells["nodes"]


def quasi_newton_update_diagonal_blocks(mesh):
    """Lloyd's algorithm can be though of a diagonal-only Hessian; this method
    incorporates the diagonal blocks, too.
    """
    X = mesh.node_coords

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    # Collect the diagonal blocks.
    # diagonal_blocks_orig = numpy.zeros((X.shape[0], 2, 2))

    # # First the Lloyd part.
    # diagonal_blocks_orig[:, 0, 0] += 2 * mesh.control_volumes
    # diagonal_blocks_orig[:, 1, 1] += 2 * mesh.control_volumes

    # diagonal_blocks = numpy.zeros((X.shape[0], 2, 2))
    # for edges, ce_ratios, ei_outer_ei in zip(
    #     mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    # ):
    #     for i, ce in zip(edges, ce_ratios):
    #         ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
    #         m = numpy.eye(2) * 0.5 * ce * numpy.dot(ei, ei)
    #         diagonal_blocks[i[0]] += m
    #         diagonal_blocks[i[1]] += m

    diagonal_blocks = numpy.zeros((X.shape[0], 2, 2))

    for edges, ce_ratios, ei_outer_ei in zip(
        mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    ):
        # m3 = -0.5 * (ce_ratios * ei_outer_ei.T).T
        for i, ce in zip(edges, ce_ratios):
            # The diagonal blocks are always positive definite if the mesh is Delaunay.
            ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
            # m = -0.5 * ce * numpy.outer(ei, ei)
            # diagonal_blocks[i[0]] += m
            # diagonal_blocks[i[1]] += m
            #
            m = 0.5 * ce * (numpy.eye(2) * numpy.dot(ei, ei) - numpy.outer(ei, ei))
            diagonal_blocks[i[0]] += m
            diagonal_blocks[i[1]] += m

    rhs = -jac_uniform(mesh).reshape(-1, 2)

    # set the boundary blocks to the identity
    diagonal_blocks[mesh.is_boundary_node] = 0.0
    diagonal_blocks[mesh.is_boundary_node, 0, 0] = 1.0
    diagonal_blocks[mesh.is_boundary_node, 1, 1] = 1.0
    rhs[mesh.is_boundary_node] = 0.0

    return numpy.linalg.solve(diagonal_blocks, rhs)


def quasi_newton_uniform_blocks(points, cells, *args, **kwargs):
    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += quasi_newton_update_diagonal_blocks(mesh)
        return x[mesh.is_interior_node]

    ghosted_mesh = GhostedMesh(points, cells)

    runner(
        get_new_points,
        ghosted_mesh.mesh,
        *args,
        **kwargs,
        straighten_out=lambda mesh: ghosted_mesh.straighten_out(),
        # get_stats_mesh=lambda mesh: ghosted_mesh.get_stats_mesh(),
    )

    # mesh = ghosted_mesh.get_stats_mesh()
    mesh = ghosted_mesh.mesh
    return mesh.node_coords, mesh.cells["nodes"]


def quasi_newton_update_full(mesh, omega):
    X = mesh.node_coords

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    i_boundary = numpy.where(mesh.is_boundary_node)[0]

    # create approximate Hessian
    row_idx = []
    col_idx = []
    vals = []
    for edges, ce_ratios, ei_outer_ei in zip(
        mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    ):
        # m3 = -0.5 * (ce_ratios * ei_outer_ei.T).T
        for edge, ce in zip(edges, ce_ratios):
            # The diagonal blocks are always positive definite if the mesh is Delaunay.
            i = edge
            ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
            ei_outer_ei = numpy.outer(ei, ei)
            # By reducing this factor, the method can be made more robust. Not sure if
            # necessary though; simply use the block-diagonal preconditioning for
            # robustness, and this one here for the last steps.
            m = -0.5 * ce * ei_outer_ei
            # (i0, i0) block
            row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
            col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
            # (i1, i1) block
            row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
            col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
            # Scale the off-diagonal blocks with some factor. If omega == 1, this is the
            # Hessian. Unfortunately, it seems that Newton domain of convergence is
            # really small. The relaxation makes the method more stable.
            m *= omega
            # (i0, i1) block
            row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
            col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
            # (i1, i0) block
            row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
            col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]

    # add diagonal
    for k, control_volume in enumerate(mesh.control_volumes):
        row_idx += [2 * k, 2 * k + 1]
        col_idx += [2 * k, 2 * k + 1]
        vals += [2 * control_volume, 2 * control_volume]

    n = mesh.control_volumes.shape[0]
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


def quasi_newton_uniform_full(points, cells, *args, omega=1.0, **kwargs):
    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += quasi_newton_update_full(mesh, omega)
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

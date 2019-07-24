# -*- coding: utf-8 -*-
#
import copy

import numpy

from meshplex import MeshTri


class GhostedMesh(MeshTri):
    def __init__(self, points, cells):
        # Add ghost points and cells for boundary facets
        msh = MeshTri(points, cells)
        ghosts = []
        ghost_cells = []
        k = points.shape[0]
        for i in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
            bf = msh.is_boundary_facet[i[0]]
            c = msh.cells["nodes"][bf].T
            ghosts.append(c[i])
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

        ghosts = numpy.concatenate(ghosts, axis=1)
        assert ghosts.shape[1] == self.num_boundary_cells

        self.num_original_points = points.shape[0]
        points = numpy.concatenate(
            [points, numpy.zeros((self.num_boundary_cells, points.shape[1]))]
        )
        self.num_original_cells = cells.shape[0]
        cells = numpy.concatenate([cells, *ghost_cells])

        # Cache some values for the ghost reflection; those values never change
        self.p1 = points[ghosts[1]]
        self.mirror_edge = points[ghosts[2]] - self.p1
        self.beta = numpy.einsum("ij, ij->i", self.mirror_edge, self.mirror_edge)

        self.ghost_mirror = ghosts[0]

        # Set ghost points
        points[self.is_ghost_point] = self.reflect_ghost(points[self.ghost_mirror])

        # Create new mesh, remember the pseudo-boundary edges
        super().__init__(points, cells)

        self.create_edges()
        # Get the first edge in the ghost cells. (The first point is the ghost point,
        # and opposite edges have the same index.)
        self.ghost_edge_gids = self.cells["edges"][is_ghost_cell, 0]
        self.original_edges_nodes = self.edges["nodes"][self.ghost_edge_gids]

        self.update_ghost_mirrors()
        return

    def get_flip_ghost_edges(self):
        # unflip boundary edges
        new_edges_nodes = self.edges["nodes"][self.ghost_edge_gids]
        # Either both nodes are equal or both are unequal; it's enough to check just the
        # first column.
        flip_edge_gids = self.ghost_edge_gids[
            self.original_edges_nodes[:, 0] != new_edges_nodes[:, 0]
        ]
        # Assert that this is actually an interior edge in the ghosted mesh, and get the
        # index into the interior edge array of the mesh.
        num_adjacent_cells, flip_interior_edge_idx = self.edge_gid_to_edge_list[
            flip_edge_gids
        ].T
        assert numpy.all(num_adjacent_cells == 2)
        return flip_interior_edge_idx

    def update_ghost_mirrors(self):
        is_ghost_cell = numpy.any(self.is_ghost_point[self.cells["nodes"]], axis=1)
        ghost_cells = self.cells["nodes"][is_ghost_cell]

        # Check which of the ghost cells
        is_outside_ghost_cell = numpy.all(self.is_boundary_node[ghost_cells], axis=1)

        # TODO If the cell is not a purely "outside ghost" cell (which are composed of
        # two boundary nodes and a ghost node), don't update the mirror. Just make sure
        # that the ghost_mirror is still connected to the ghost point by at least one
        # cell.

        # Now let's look at the ghost points which belong to only one cell. We need to
        # find the cell on the other side and in there the point opposite of the ghost
        # edge.
        # Find all cells which have exactly two boundary cells; they are the opposites.
        is_opposite_cell = (
            numpy.sum(self.is_boundary_node[self.cells["nodes"]], axis=1) == 2
        ) & ~is_ghost_cell
        assert numpy.sum(is_opposite_cell) == numpy.sum(is_outside_ghost_cell)
        # Now we need to match the opposite cells with the ghost cells by checking which
        # pairs have two nodes in common.
        opposite_cells = self.cells["nodes"][is_opposite_cell]
        edges0 = numpy.sort(
            opposite_cells[self.is_boundary_node[opposite_cells]].reshape(-1, 2), axis=1
        )
        outside_ghost_cells = ghost_cells[is_outside_ghost_cell]
        edges1 = numpy.sort(
            outside_ghost_cells[
                self.is_boundary_node[outside_ghost_cells]
                & ~self.is_ghost_point[outside_ghost_cells]
            ].reshape(-1, 2),
            axis=1,
        )
        assert len(edges0) == len(edges1)
        # For faster variants see <https://stackoverflow.com/q/57184214/353337>
        i = [numpy.where(numpy.all(edge == edges0, axis=1))[0][0] for edge in edges1]
        assert numpy.all(edges0[i] == edges1)
        # Find which ghost_mirrors need updating
        ghost_cells = ghost_cells[is_outside_ghost_cell]
        idx = ghost_cells[self.is_ghost_point[ghost_cells]] - (
            self.node_coords.shape[0] - self.ghost_mirror.shape[0]
        )
        # The ghost mirror is the non-boundary point in the opposite cell.
        # sort the opposite cells
        opposite_cells = opposite_cells[i]
        self.ghost_mirror[idx] = opposite_cells[~self.is_boundary_node[opposite_cells]]

        # make sure that no ghost mirror is a ghost point; that'd be a bug
        assert not numpy.any(
            self.is_ghost_point[self.ghost_mirror]
        ), self.is_ghost_point[self.ghost_mirror]

        # update point values
        self.node_coords[self.is_ghost_point] = self.reflect_ghost(
            self.node_coords[self.ghost_mirror]
        )
        # updating _all_ values is a bit overkill actually; we only need to update the
        # values in the ghost cells
        self.update_values()
        return

    def get_unghosted_mesh(self):
        # Make deep copy to avoid influencing the actual mesh
        mesh2 = copy.deepcopy(self)
        mesh2.flip_interior_edges(self.get_flip_ghost_edges())

        # remove ghost points and cells
        points = mesh2.node_coords[: self.num_original_points]
        has_ghost_point = numpy.any(
            mesh2.cells["nodes"] >= self.num_original_points, axis=1
        )
        cells = mesh2.cells["nodes"][~has_ghost_point]
        return MeshTri(points, cells)

    def flip_until_delaunay(self):
        # print()
        # print("BB")
        # self.show(
        #     # show_node_numbers=True, show_cell_numbers=True
        #     )
        super().flip_until_delaunay()
        # print("CC")
        # self.show(
        #     # show_node_numbers=True, show_cell_numbers=True
        #     )
        self.update_ghost_mirrors()
        # print("DD")
        # self.show(
        #     # show_node_numbers=True, show_cell_numbers=True
        #     )
        return

    def reflect_ghost(self, p0):
        """This method returns the ghost point p0', namely p0 reflected along the edge
        p1--p2.

                p0
              _/| \\__
            _/  |    \\__
           /    |       \\
          p1----|q-------p2
          \\_   |     __/
            \\_ |  __/
              \\| /
               p0'

        """
        # Instead of self.p1, one could take any point on the line p1--p2.
        dist = self.p1 - p0
        alpha = numpy.einsum("ij, ij->i", dist, self.mirror_edge)
        # q is sits at the perpendicular intersection of the reflection
        q = dist - (alpha / self.beta)[:, None] * self.mirror_edge
        return p0 + 2 * q

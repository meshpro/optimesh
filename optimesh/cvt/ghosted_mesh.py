# -*- coding: utf-8 -*-
#
import copy

import numpy

from meshplex import MeshTri


class GhostedMesh(MeshTri):
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

        # Cache some values for the ghost reflection; those values never change
        self.p1 = points[self.ghost_mirror[1]]
        self.mirror_edge = points[self.ghost_mirror[2]] - self.p1
        self.beta = numpy.einsum("ij, ij->i", self.mirror_edge, self.mirror_edge)

        # Set ghost points
        points[self.is_ghost_point] = self.reflect_ghost(points[self.ghost_mirror[0]])

        # Create new mesh, remember the pseudo-boundary edges
        super(GhostedMesh, self).__init__(points, cells)

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
        # The ghost mirror facet points ghost_mirror[1], ghost_mirror[2] are on the
        # boundary and never change in any way. The point that is mirrored,
        # ghost_mirror[0], however, does move and may after a facet flip even refer to
        # an entirely different point. Find out which.
        ghost_mirror0 = numpy.zeros(self.num_boundary_cells, dtype=int)

        new_edges_nodes = self.edges["nodes"][self.ghost_edge_gids]
        has_flipped = self.original_edges_nodes[:, 0] != new_edges_nodes[:, 0]

        # In the beginning, the ghost points are appended to the points array and hence
        # have higher GIDs than all other points. Since the edges["nodes"] are sorted,
        # the second entry must be the ghost point.
        assert numpy.all(self.is_ghost_point[new_edges_nodes[has_flipped, 1]]), (
            "A ghost edge is flipped, but does not contain the ghost point. Try "
            "applying some steps of a more robust method first."
        )
        # The first point is the one at the other end of the flipped edge.
        ghost_mirror0[has_flipped] = new_edges_nodes[has_flipped, 0]

        # Now let's look at the ghost points whose edge has _not_ flipped. We need to
        # find the cell on the other side and in there the point opposite of the ghost
        # edge.
        num_adjacent_cells, interior_edge_idx = self.edge_gid_to_edge_list[
            self.ghost_edge_gids
        ][~has_flipped].T
        assert numpy.all(num_adjacent_cells == 2)
        #
        adj_cells = self.edges_cells[2][interior_edge_idx]
        is_1st = adj_cells[:, 0] == self.ghost_cell_gids[~has_flipped]
        is_2nd = adj_cells[:, 1] == self.ghost_cell_gids[~has_flipped]
        assert numpy.all(numpy.logical_xor(is_1st, is_2nd))
        #
        opposite_cell_id = numpy.empty(adj_cells.shape[0], dtype=int)
        opposite_cell_id[is_1st] = adj_cells[is_1st, 1]
        opposite_cell_id[is_2nd] = adj_cells[is_2nd, 0]
        # Now find the cell opposite of the ghost edge in the opposite cell.
        eq = numpy.array(
            [
                self.cells["edges"][opposite_cell_id, k]
                == self.ghost_edge_gids[~has_flipped]
                for k in range(self.cells["edges"].shape[1])
            ]
        )
        assert numpy.all(numpy.sum(eq, axis=0) == 1)
        opposite_node_id = numpy.empty(eq.shape[1], dtype=int)
        cn = self.cells["nodes"][opposite_cell_id]
        for k in range(eq.shape[0]):
            opposite_node_id[eq[k]] = cn[eq[k], k]
        # Set in mirrors
        ghost_mirror0[~has_flipped] = opposite_node_id

        self.ghost_mirror[0] = ghost_mirror0
        self.mirrors = ghost_mirror0  # only for backwards compatibility

        # update point values
        x = self.node_coords
        x[self.is_ghost_point] = self.reflect_ghost(x[self.ghost_mirror[0]])
        # updating _all_ values is a bit overkill actually; we only need to update the
        # values in the ghost cells
        self.update_values()
        return

    def get_unghosted_mesh(self):
        # Make deep copy to avoid influencing the actual mesh
        mesh2 = copy.deepcopy(self)
        mesh2.flip_interior_edges(self.get_flip_ghost_edges())
        # remove ghost cells
        # TODO this is too crude; sometimes the wrong cells are cut
        points = mesh2.node_coords[: self.num_original_points]
        cells = mesh2.cells["nodes"][: self.num_original_cells]
        return MeshTri(points, cells)

    def flip_until_delaunay(self):
        super(GhostedMesh, self).flip_until_delaunay()
        self.update_ghost_mirrors()
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

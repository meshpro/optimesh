# -*- coding: utf-8 -*-
#
from __future__ import print_function

import fastfunc
from meshplex import MeshTri
import numpy

from .helpers import runner


def fixed_point(points, cells, *args, **kwargs):
    """Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    """

    def get_new_points(mesh):
        # move interior points into average of their neighbors
        num_neighbors = numpy.zeros(len(mesh.node_coords), dtype=int)
        idx = mesh.edges["nodes"]
        fastfunc.add.at(num_neighbors, idx, numpy.ones(idx.shape, dtype=int))

        new_points = numpy.zeros(mesh.node_coords.shape)
        fastfunc.add.at(new_points, idx[:, 0], mesh.node_coords[idx[:, 1]])
        fastfunc.add.at(new_points, idx[:, 1], mesh.node_coords[idx[:, 0]])

        idx = mesh.is_interior_node
        new_points = (new_points[idx].T / num_neighbors[idx]).T
        return new_points

    mesh = MeshTri(points, cells)
    runner(get_new_points, mesh, *args, **kwargs)
    return mesh.node_coords, mesh.cells["nodes"]

# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy

from .helpers import runner


def laplace(*args, **kwargs):
    """Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    """
    def get_new_points(mesh):
        # move interior points into average of their neighbors
        # <https://stackoverflow.com/a/43096495/353337>
        # num_neighbors = numpy.bincount(mesh.edges['nodes'].flat)
        #
        boundary_verts = mesh.get_boundary_vertices()

        num_neighbors = numpy.zeros(mesh.node_coords.shape[0], dtype=int)
        new_points = numpy.zeros(mesh.node_coords.shape)
        for edge in mesh.edges["nodes"]:
            num_neighbors[edge[0]] += 1
            num_neighbors[edge[1]] += 1
            new_points[edge[0]] += mesh.node_coords[edge[1]]
            new_points[edge[1]] += mesh.node_coords[edge[0]]
        new_points = (new_points.T / num_neighbors).T

        # Keep the boundary vertices in place
        new_points[boundary_verts] = mesh.node_coords[boundary_verts]
        return new_points

    return runner(get_new_points, *args, **kwargs)

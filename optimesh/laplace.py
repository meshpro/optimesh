import numpy
from meshplex import MeshTri

from .helpers import runner


def fixed_point(points, cells, *args, **kwargs):
    """Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    """

    def get_new_points(mesh):
        # move interior points into average of their neighbors

        # old way:
        # num_neighbors = numpy.zeros(n, dtype=int)
        # numpy.add.at(num_neighbors, idx, numpy.ones(idx.shape, dtype=int))
        # new_points = numpy.zeros(mesh.node_coords.shape)
        # numpy.add.at(new_points, idx[:, 0], mesh.node_coords[idx[:, 1]])
        # numpy.add.at(new_points, idx[:, 1], mesh.node_coords[idx[:, 0]])

        n = mesh.node_coords.shape[0]
        idx = mesh.edges["nodes"]
        num_neighbors = numpy.bincount(idx.reshape(-1), minlength=n)

        new_points = numpy.zeros(mesh.node_coords.shape)
        vals = mesh.node_coords[idx[:, 1]].T
        new_points += numpy.array(
            [numpy.bincount(idx[:, 0], val, minlength=n) for val in vals]
        ).T
        vals = mesh.node_coords[idx[:, 0]].T
        new_points += numpy.array(
            [numpy.bincount(idx[:, 1], val, minlength=n) for val in vals]
        ).T
        new_points /= num_neighbors[:, None]

        # reset boundary nodes
        idx = mesh.is_boundary_node
        new_points[idx] = mesh.node_coords[idx]
        return new_points

    mesh = MeshTri(points, cells)
    runner(get_new_points, mesh, *args, **kwargs)
    return mesh.node_coords, mesh.cells["nodes"]

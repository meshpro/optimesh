import numpy


def get_new_points(mesh):
    """Perform one steps of Laplacian smoothing to the mesh, i.e., move each interior
    vertex to the arithmetic average of its neighboring points.
    """
    # move interior points into average of their neighbors

    # old way:
    # num_neighbors = numpy.zeros(n, dtype=int)
    # numpy.add.at(num_neighbors, idx, numpy.ones(idx.shape, dtype=int))
    # new_points = numpy.zeros(mesh.points.shape)
    # numpy.add.at(new_points, idx[:, 0], mesh.points[idx[:, 1]])
    # numpy.add.at(new_points, idx[:, 1], mesh.points[idx[:, 0]])

    n = mesh.points.shape[0]
    idx = mesh.edges["points"]
    num_neighbors = numpy.bincount(idx.reshape(-1), minlength=n)

    new_points = numpy.zeros(mesh.points.shape)
    vals = mesh.points[idx[:, 1]].T
    new_points += numpy.array(
        [numpy.bincount(idx[:, 0], val, minlength=n) for val in vals]
    ).T
    vals = mesh.points[idx[:, 0]].T
    new_points += numpy.array(
        [numpy.bincount(idx[:, 1], val, minlength=n) for val in vals]
    ).T
    new_points /= num_neighbors[:, None]

    # reset boundary points
    idx = mesh.is_boundary_point
    new_points[idx] = mesh.points[idx]
    return new_points

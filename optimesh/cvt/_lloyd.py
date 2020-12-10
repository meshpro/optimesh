import numpy
from meshplex import MeshTri

from ..helpers import runner


def get_new_points(mesh, boundary_step=None):
    # Exclude all cells which have a too negative covolume-edgelength ratio. This is
    # necessary to prevent points to be dragged outside of the domain by very flat
    # cells on the boundary.
    # There are other possible heuristics too. For example, one could restrict the
    # mask to cells at or near the boundary.
    mask = numpy.any(mesh.ce_ratios < -0.5, axis=0)

    X = mesh.get_control_volume_centroids(cell_mask=mask)

    # When using a cell mask, it can happen that some points don't get any
    # contribution at all because they are adjacent only to masked cells. Reset
    # those, too.
    idx = numpy.any(numpy.isnan(X), axis=1)
    X[idx] = mesh.points[idx]

    if boundary_step is None:
        # Reset boundary points to their original positions.
        idx = mesh.is_boundary_point
        X[idx] = mesh.points[idx]
    else:
        # Move all boundary points back to the boundary.
        idx = mesh.is_boundary_point
        X[idx] = boundary_step(X[idx].T).T

    return X


def quasi_newton_uniform_lloyd(points, cells, *args, boundary_step=None, **kwargs):
    """Lloyd's algorithm.
    Check out

    Xiao Xiao,
    Over-Relaxation Lloyd Method For Computing Centroidal Voronoi Tessellations,
    Master's thesis, Jan. 2010,
    University of South Carolina,
    <https://scholarcommons.sc.edu/etd/295/>

    for use of the relaxation paramter. (omega=2 is suggested.)

    Everything above omega=2 can lead to flickering, i.e., rapidly alternating updates
    and bad meshes.
    """

    mesh = MeshTri(points, cells)

    method_name = "Lloyd's algorithm"
    runner(get_new_points, mesh, *args, **kwargs, method_name=method_name)

    return mesh.points, mesh.cells["points"]

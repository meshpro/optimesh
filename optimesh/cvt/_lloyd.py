import numpy

from meshplex import MeshTri

from ..helpers import runner


def quasi_newton_uniform_lloyd(points, cells, *args, boundary=None, **kwargs):
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

    def get_new_points(mesh):
        # Exclude all cells which have a too negative covolume-edgelength ratio. This is
        # necessary to prevent nodes to be dragged outside of the domain by very flat
        # cells on the boundary.
        # There are other possible heuristics too. For example, one could restrict the
        # mask to cells at or near the boundary.
        mask = numpy.any(mesh.ce_ratios < -0.5, axis=0)

        X = mesh.get_control_volume_centroids(cell_mask=mask)

        # When using a cell mask, it can happen that some nodes don't get any
        # contribution at all because they are adjacent only to masked cells. Reset
        # those, too.
        idx = numpy.any(numpy.isnan(X), axis=1)
        X[idx] = mesh.node_coords[idx]

        if boundary is None:
            # Reset boundary points to their original positions.
            idx = mesh.is_boundary_node
            X[idx] = mesh.node_coords[idx]
        else:
            # Move all boundary nodes back to the boundary.
            # import meshplex

            # mesh = meshplex.MeshTri(X, cells)
            # mesh.show()

            idx = mesh.is_boundary_node
            X[idx] = boundary.boundary_step(X[idx].T).T

            # mesh = meshplex.MeshTri(X, cells)
            # mesh.show()
        return X

    mesh = MeshTri(points, cells)

    method_name = "Lloyd's algorithm"
    runner(get_new_points, mesh, *args, **kwargs, method_name=method_name)

    return mesh.node_coords, mesh.cells["nodes"]

import numpy as np


def get_new_points(mesh):
    """Lloyd's algorithm.
    Check out

    Xiao Xiao,
    Over-Relaxation Lloyd Method For Computing Centroidal Voronoi Tessellations,
    Master's thesis, Jan. 2010,
    University of South Carolina,
    <https://scholarcommons.sc.edu/etd/295/>

    for use of the relaxation parameter. (omega=2 is suggested.)

    Everything above omega=2 can lead to flickering, i.e., rapidly alternating updates
    and bad meshes.
    """
    # Exclude all cells which have a too negative covolume-edgelength ratio. This is
    # necessary to prevent points to be dragged outside of the domain by very flat cells
    # on the boundary.
    # There are other possible heuristics too. For example, one could restrict the mask
    # to cells at or near the boundary.
    ce2 = mesh.ce_ratios.reshape(-1, mesh.ce_ratios.shape[-1])
    cell_mask = np.any(ce2 < -0.5, axis=0)

    X = mesh.get_control_volume_centroids(cell_mask)

    # When using a cell mask, it can happen that some points don't get any contribution
    # at all because they are adjacent only to masked cells. Reset those, too.
    x = X.reshape(X.shape[0], -1)
    idx = np.any(np.isnan(x), axis=1)
    X[idx] = mesh.points[idx]
    return X

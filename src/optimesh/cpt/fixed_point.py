import meshplex
import numpy as np

from ..helpers import get_new_points_averaged


def get_new_points(mesh: meshplex.Mesh) -> np.ndarray:
    """Idea:
    Move interior mesh points into the weighted averages of the centroids (barycenters)
    of their adjacent cells.
    """
    return get_new_points_averaged(mesh, mesh.cell_barycenters, mesh.cell_volumes)

"""Uniform density variant.
"""
from ..helpers import get_new_points_averaged


def get_new_points(mesh):
    """Idea:
    Move interior mesh points into the weighted averages of the circumcenters
    of their adjacent cells. (Except on boundary cells; use barycenters there.)
    """
    # Get circumcenters everywhere except at cells adjacent to the boundary;
    # barycenters there.
    cc = mesh.cell_circumcenters
    bc = mesh.cell_barycenters
    # Find all cells with a boundary edge
    cc[mesh.is_boundary_cell] = bc[mesh.is_boundary_cell]
    X = get_new_points_averaged(mesh, cc, mesh.cell_volumes)
    return X


# Not working as density-preserving method. TODO investigate
# def get_new_points(mesh):
#     """Idea:
#     Move interior mesh points into the weighted averages of the circumcenters
#     of their adjacent cells.
#     """
#     # Get circumcenters everywhere except at cells adjacent to the boundary;
#     # barycenters there. The reason is that points near the boundary would be
#     # "sucked" out of the domain if the boundary cell is very flat, i.e., its
#     # circumcenter is very far outside of the domain.
#     # This heuristic also applies to cells _near_ the boundary though, and, if
#     # constructed maliciously, any mesh. Hence, this method can break down. A better
#     # approach is to use barycenters for all cells which are rather flat.
#     cc = mesh.cell_circumcenters.copy()
#     # Find all cells with a boundary edge
#     is_boundary_cell = (
#         np.sum(mesh.is_boundary_point[mesh.cells["points"]], axis=1) == 2
#     )
#     cc[is_boundary_cell] = mesh.cell_barycenters[is_boundary_cell]
#     X = get_new_points_averaged(mesh, cc)
#     return X

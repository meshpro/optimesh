# -*- coding: utf-8 -*-
#
from .ghosted_mesh import GhostedMesh
from .helpers import jac_uniform
from ..helpers import runner


def quasi_newton_uniform_lloyd(points, cells, *args, omega=1.0, **kwargs):
    """Relaxed Lloyd's algorithm. omega=1 leads to Lloyd's algorithm, overrelaxation
    omega=2 gives good results. Check out

    Xiao Xiao,
    Over-Relaxation Lloyd Method For Computing Centroidal Voronoi Tessellations,
    Master's thesis,
    <https://scholarcommons.sc.edu/etd/295/>.

    Everything above omega=2 can lead to flickering, i.e., rapidly alternating updates
    and bad meshes.
    """

    def get_new_points(mesh):
        x = (
            mesh.node_coords
            - omega / 2 * jac_uniform(mesh) / mesh.control_volumes[:, None]
        )
        # update boundary and ghosts
        idx = mesh.is_boundary_node & ~ghosted_mesh.is_ghost_point
        x[idx] = mesh.node_coords[idx]
        x[ghosted_mesh.is_ghost_point] = ghosted_mesh.reflect_ghost(
            x[ghosted_mesh.mirrors]
        )
        return x

    ghosted_mesh = GhostedMesh(points, cells)

    method_name = "Lloyd's algorithm"
    if abs(omega - 1.0) > 1.0e-10:
        method_name += ", relaxation parameter {}".format(omega)

    runner(
        get_new_points,
        ghosted_mesh,
        *args,
        **kwargs,
        update_topology=lambda mesh: ghosted_mesh.update_topology(),
        get_stats_mesh=lambda mesh: ghosted_mesh.get_unghosted_mesh(),
        method_name=method_name,
    )

    mesh = ghosted_mesh.get_unghosted_mesh()
    return mesh.node_coords, mesh.cells["nodes"]


# def fixed_point_uniform(points, cells, *args, **kwargs):
#     """Lloyd's algorithm.
#     """
#     assert points.shape[1] == 2
#
#     def get_new_points(mesh):
#         return mesh.control_volume_centroids[mesh.is_interior_node]
#
#     ghosted_mesh = GhostedMesh(points, cells)
#
#     runner(
#         get_new_points,
#         ghosted_mesh,
#         *args,
#         **kwargs,
#         update_topology=lambda mesh: ghosted_mesh.update_topology(),
#         # get_stats_mesh=lambda mesh: ghosted_mesh.get_unghosted_mesh(),
#     )
#
#     mesh = ghosted_mesh.get_unghosted_mesh()
#     return mesh.node_coords, mesh.cells["nodes"]

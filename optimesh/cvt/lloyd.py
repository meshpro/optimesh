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
        return x[mesh.is_interior_node]

    ghosted_mesh = GhostedMesh(points, cells)

    runner(
        get_new_points,
        ghosted_mesh.mesh,
        *args,
        **kwargs,
        straighten_out=lambda mesh: ghosted_mesh.straighten_out(),
        # get_stats_mesh=lambda mesh: ghosted_mesh.get_stats_mesh(),
    )

    mesh = ghosted_mesh.get_stats_mesh()
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
#         ghosted_mesh.mesh,
#         *args,
#         **kwargs,
#         straighten_out=lambda mesh: ghosted_mesh.straighten_out(),
#         # get_stats_mesh=lambda mesh: ghosted_mesh.get_stats_mesh(),
#     )
#
#     mesh = ghosted_mesh.get_stats_mesh()
#     return mesh.node_coords, mesh.cells["nodes"]

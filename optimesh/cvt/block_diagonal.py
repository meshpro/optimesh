# -*- coding: utf-8 -*-
#
import fastfunc
import numpy

from .ghosted_mesh import GhostedMesh
from .helpers import jac_uniform

from ..helpers import runner


def quasi_newton_uniform_blocks(points, cells, *args, **kwargs):
    """Lloyd's algorithm can be though of a diagonal-only Hessian; this method
    incorporates the diagonal blocks, too.
    """

    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += update(mesh)
        # update ghosts
        x[ghosted_mesh.is_ghost_point] = ghosted_mesh.reflect_ghost(
            x[ghosted_mesh.mirrors]
        )
        return x

    ghosted_mesh = GhostedMesh(points, cells)

    runner(
        get_new_points,
        ghosted_mesh,
        *args,
        **kwargs,
        update_topology=lambda mesh: ghosted_mesh.update_topology(),
        get_stats_mesh=lambda mesh: ghosted_mesh.get_unghosted_mesh(),
    )

    mesh = ghosted_mesh.get_unghosted_mesh()
    return mesh.node_coords, mesh.cells["nodes"]


def update(mesh):
    X = mesh.node_coords

    # Collect the diagonal blocks.
    diagonal_blocks = numpy.zeros((X.shape[0], X.shape[1], X.shape[1]))

    # First the Lloyd part.
    for k in range(X.shape[1]):
        diagonal_blocks[:, k, k] += 2 * mesh.control_volumes

    ei_outer_ei = numpy.einsum(
        "ijk, ijl->ijkl", mesh.half_edge_coords, mesh.half_edge_coords
    )

    # Without adding the control volumes above, the update would be
    # ```
    # m = 0.5 * ce * (numpy.eye(X.shape[1]) * numpy.dot(ei, ei) - ei_outer_ei)
    # ```
    # instead of
    # ```
    # m = -0.5 * ce * ei_outer_ei
    # ```
    # This makes clear why the diagonal blocks are always positive definite if the mesh
    # is Delaunay.
    M = -0.5 * ei_outer_ei * mesh.ce_ratios[..., None, None]

    fastfunc.add.at(diagonal_blocks, mesh.idx_hierarchy[0], M)
    fastfunc.add.at(diagonal_blocks, mesh.idx_hierarchy[1], M)

    rhs = -jac_uniform(mesh)

    # set the boundary blocks to the identity
    diagonal_blocks[mesh.is_boundary_node] = 0.0
    for k in range(X.shape[1]):
        diagonal_blocks[mesh.is_boundary_node, k, k] = 1.0
    rhs[mesh.is_boundary_node] = 0.0

    return numpy.linalg.solve(diagonal_blocks, rhs)

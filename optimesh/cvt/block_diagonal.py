# -*- coding: utf-8 -*-
#
import numpy

from .ghosted_mesh import GhostedMesh
from .helpers import jac_uniform

from ..helpers import runner


def quasi_newton_uniform_blocks(points, cells, *args, **kwargs):
    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += quasi_newton_update_diagonal_blocks(mesh)
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
    # mesh = ghosted_mesh.mesh
    return mesh.node_coords, mesh.cells["nodes"]


def quasi_newton_update_diagonal_blocks(mesh):
    """Lloyd's algorithm can be though of a diagonal-only Hessian; this method
    incorporates the diagonal blocks, too.
    """
    X = mesh.node_coords

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    # Collect the diagonal blocks.
    # diagonal_blocks_orig = numpy.zeros((X.shape[0], 2, 2))

    # # First the Lloyd part.
    # diagonal_blocks_orig[:, 0, 0] += 2 * mesh.control_volumes
    # diagonal_blocks_orig[:, 1, 1] += 2 * mesh.control_volumes

    # diagonal_blocks = numpy.zeros((X.shape[0], 2, 2))
    # for edges, ce_ratios, ei_outer_ei in zip(
    #     mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    # ):
    #     for i, ce in zip(edges, ce_ratios):
    #         ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
    #         m = numpy.eye(2) * 0.5 * ce * numpy.dot(ei, ei)
    #         diagonal_blocks[i[0]] += m
    #         diagonal_blocks[i[1]] += m

    diagonal_blocks = numpy.zeros((X.shape[0], 2, 2))

    for edges, ce_ratios, ei_outer_ei in zip(
        mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    ):
        # m3 = -0.5 * (ce_ratios * ei_outer_ei.T).T
        for i, ce in zip(edges, ce_ratios):
            # The diagonal blocks are always positive definite if the mesh is Delaunay.
            ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
            # m = -0.5 * ce * numpy.outer(ei, ei)
            # diagonal_blocks[i[0]] += m
            # diagonal_blocks[i[1]] += m
            #
            m = 0.5 * ce * (numpy.eye(2) * numpy.dot(ei, ei) - numpy.outer(ei, ei))
            diagonal_blocks[i[0]] += m
            diagonal_blocks[i[1]] += m

    rhs = -jac_uniform(mesh).reshape(-1, 2)

    # set the boundary blocks to the identity
    diagonal_blocks[mesh.is_boundary_node] = 0.0
    diagonal_blocks[mesh.is_boundary_node, 0, 0] = 1.0
    diagonal_blocks[mesh.is_boundary_node, 1, 1] = 1.0
    rhs[mesh.is_boundary_node] = 0.0

    return numpy.linalg.solve(diagonal_blocks, rhs)

import numpy

import fastfunc
from meshplex import MeshTri

from ..helpers import runner

# from .ghosted_mesh import GhostedMesh
from .helpers import jac_uniform


def quasi_newton_uniform_blocks(points, cells, *args, **kwargs):
    """Lloyd's algorithm can be though of a diagonal-only Hessian; this method
    incorporates the diagonal blocks, too. It's almost as cheap but performs better.
    """

    def get_new_points(mesh):
        # Exclude all cells which have a too negative covolume-edgelength ratio. This is
        # necessary to prevent nodes to be dragged outside of the domain by very flat
        # cells on the boundary.
        # There are other possible heuristics too. For example, one could restrict the
        # mask to cells at or near the boundary.
        mask = numpy.any(mesh.ce_ratios < -0.5, axis=0)

        X = mesh.node_coords.copy()
        # Collect the diagonal blocks.
        diagonal_blocks = numpy.zeros((X.shape[0], X.shape[1], X.shape[1]))

        # First the Lloyd part.
        cv = mesh.get_control_volumes(cell_mask=mask)
        for k in range(X.shape[1]):
            diagonal_blocks[:, k, k] += 2 * cv

        hec = mesh.half_edge_coords[:, ~mask]
        ei_outer_ei = numpy.einsum("ijk, ijl->ijkl", hec, hec)

        # Without adding the control volumes above, the update would be
        # ```
        # m = 0.5 * ce * (numpy.eye(X.shape[1]) * numpy.dot(ei, ei) - ei_outer_ei)
        # ```
        # instead of
        # ```
        # m = -0.5 * ce * ei_outer_ei
        # ```
        # This makes clear why the diagonal blocks are always positive definite if the
        # mesh is Delaunay.
        M = -0.5 * ei_outer_ei * mesh.ce_ratios[:, ~mask, None, None]

        fastfunc.add.at(diagonal_blocks, mesh.idx_hierarchy[0][:, ~mask], M)
        fastfunc.add.at(diagonal_blocks, mesh.idx_hierarchy[1][:, ~mask], M)

        rhs = -jac_uniform(mesh, mask)

        # Set the boundary blocks to the identity. When using a cell mask, it can
        # happen that some nodes don't get any contribution at all because they are
        # adjacent only to masked cells. Reset those, too.
        idx = numpy.any(numpy.isnan(rhs), axis=1) | mesh.is_boundary_node
        diagonal_blocks[idx] = 0.0
        for k in range(X.shape[1]):
            diagonal_blocks[idx, k, k] = 1.0
        rhs[idx] = 0.0

        X += numpy.linalg.solve(diagonal_blocks, rhs)
        return X

    mesh = MeshTri(points, cells)

    runner(
        get_new_points,
        mesh,
        *args,
        **kwargs,
        method_name="Centroidal Voronoi Tesselation (CVT), uniform density, block-diagonal variant"
    )
    return mesh.node_coords, mesh.cells["nodes"]

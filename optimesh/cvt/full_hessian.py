# -*- coding: utf-8 -*-
#
import numpy
import scipy.sparse

from .ghosted_mesh import GhostedMesh
from .helpers import jac_uniform

from ..helpers import runner


def quasi_newton_uniform_full(points, cells, *args, omega=1.0, **kwargs):
    def get_new_points(mesh):
        # TODO need copy?
        x = mesh.node_coords.copy()
        x += update(mesh, omega)
        return x[mesh.is_interior_node]

    ghosted_mesh = GhostedMesh(points, cells)

    runner(
        get_new_points,
        ghosted_mesh.mesh,
        *args,
        **kwargs,
        straighten_out=lambda mesh: ghosted_mesh.straighten_out(),
        get_stats_mesh=lambda mesh: ghosted_mesh.get_stats_mesh(),
    )

    mesh = ghosted_mesh.get_stats_mesh()
    # mesh = ghosted_mesh.mesh
    return mesh.node_coords, mesh.cells["nodes"]


def update(mesh, omega):
    X = mesh.node_coords

    # TODO remove this assertion and test
    # flat mesh
    assert X.shape[1] == 2

    i_boundary = numpy.where(mesh.is_boundary_node)[0]

    # create approximate Hessian
    row_idx = []
    col_idx = []
    vals = []
    for edges, ce_ratios, ei_outer_ei in zip(
        mesh.idx_hierarchy.T, mesh.ce_ratios.T, numpy.moveaxis(mesh.ei_outer_ei, 0, 1)
    ):
        # m3 = -0.5 * (ce_ratios * ei_outer_ei.T).T
        for edge, ce in zip(edges, ce_ratios):
            # The diagonal blocks are always positive definite if the mesh is Delaunay.
            i = edge
            ei = mesh.node_coords[i[1]] - mesh.node_coords[i[0]]
            ei_outer_ei = numpy.outer(ei, ei)
            # By reducing this factor, the method can be made more robust. Not sure if
            # necessary though; simply use the block-diagonal preconditioning for
            # robustness, and this one here for the last steps.
            m = -0.5 * ce * ei_outer_ei
            # (i0, i0) block
            row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
            col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
            # (i1, i1) block
            row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
            col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
            # Scale the off-diagonal blocks with some factor. If omega == 1, this is the
            # Hessian. Unfortunately, it seems that Newton domain of convergence is
            # really small. The relaxation makes the method more stable.
            m *= omega
            # (i0, i1) block
            row_idx += [2 * i[0] + 0, 2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 1]
            col_idx += [2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 0, 2 * i[1] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
            # (i1, i0) block
            row_idx += [2 * i[1] + 0, 2 * i[1] + 0, 2 * i[1] + 1, 2 * i[1] + 1]
            col_idx += [2 * i[0] + 0, 2 * i[0] + 1, 2 * i[0] + 0, 2 * i[0] + 1]
            vals += [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]

    # add diagonal
    for k, control_volume in enumerate(mesh.control_volumes):
        row_idx += [2 * k, 2 * k + 1]
        col_idx += [2 * k, 2 * k + 1]
        vals += [2 * control_volume, 2 * control_volume]

    n = mesh.control_volumes.shape[0]
    matrix = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(2 * n, 2 * n))

    # Transform to CSR format for efficiency
    matrix = matrix.tocsr()

    # Apply Dirichlet conditions.
    # Set all Dirichlet rows to 0.
    for i in numpy.where(mesh.is_boundary_node)[0]:
        matrix.data[matrix.indptr[2 * i + 0] : matrix.indptr[2 * i + 0 + 1]] = 0.0
        matrix.data[matrix.indptr[2 * i + 1] : matrix.indptr[2 * i + 1 + 1]] = 0.0
    # Set the diagonal and RHS.
    d = matrix.diagonal()
    d[2 * i_boundary + 0] = 1.0
    d[2 * i_boundary + 1] = 1.0
    matrix.setdiag(d)

    rhs = -jac_uniform(mesh)
    rhs[2 * i_boundary + 0] = 0.0
    rhs[2 * i_boundary + 1] = 0.0

    # print("ok hi")
    # print(numpy.sort(numpy.linalg.eigvals(matrix.toarray())))
    # exit(1)

    out = scipy.sparse.linalg.spsolve(matrix, rhs)
    # ml = pyamg.ruge_stuben_solver(matrix)
    # out = ml.solve(rhs, tol=1.0e-10)

    return out.reshape(-1, 2)

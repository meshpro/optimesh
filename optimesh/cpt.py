# -*- coding: utf-8 -*-
#
"""
From

Long Chen, Michael Holst,
Efficient mesh optimization schemes based on Optimal Delaunay
Triangulations,
Comput. Methods Appl. Mech. Engrg. 200 (2011) 967–984,
<https://doi.org/10.1016/j.cma.2010.11.007>.
"""
from meshplex import MeshTri
import numpy
import optipy
import pyamg
import quadpy
import scipy.sparse

from .helpers import (
    runner,
    get_new_points_volume_averaged,
    get_new_points_count_averaged,
)


def energy(X, cells, uniform_density=False):
    """The CPT mesh energy is defined as

    E~ = 1/(d+1) * sum int_{omega_i} ||x - x_i||^2 rho(x) dx,

    see Chen-Holst.
    """
    dim = 2
    mesh = MeshTri(X, cells, flat_cell_correction=None)

    star_integrals = numpy.zeros(mesh.node_coords.shape[0])
    # Python loop over the cells... slow!
    for cell, cell_volume in zip(mesh.cells["nodes"], mesh.cell_volumes):
        for i in range(3):
            idx = cell[i]
            xi = mesh.node_coords[idx]
            val = quadpy.triangle.integrate(
                lambda x: (x[0] - xi[0]) ** 2 + (x[1] - xi[1]) ** 2,
                mesh.node_coords[cell],
                # Take any scheme with order 2
                quadpy.triangle.Dunavant(2),
            )
            if not uniform_density:
                # rho = 1,
                star_integrals[idx] += val
            else:
                # rho = 1 / tau_j,
                star_integrals[idx] += val / cell_volume

    return numpy.sum(star_integrals) / (dim + 1)


def jac(x):
    """The approximated Jacobian

    \partial_i E = 2/(d+1) (x_i int_{omega_i} rho(x) dx - int_{omega_i} x rho(x) dx)
    """
    dim = 2
    mesh = 1
    return


def fixed_point(*args, uniform_density=False, **kwargs):
    """Centroidal Patch Triangulation. Mimics the definition of Centroidal
    Voronoi Tessellations for which the generator and centroid of each Voronoi
    region coincide.

    Idea:
    Move interior mesh points into the weighted averages of the centroids
    (barycenters) of their adjacent cells. If a triangle cell switches
    orientation in the process, don't move quite so far.
    """
    compute_average = (
        get_new_points_volume_averaged
        if uniform_density
        else get_new_points_count_averaged
    )

    def get_new_points(mesh):
        return compute_average(mesh, mesh.get_cell_barycenters())

    return runner(get_new_points, *args, **kwargs)


def linear_solve(*args, **kwargs):
    """The `i`th entry in the CPT-energy gradient is

        \\partial E_i = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) \\int_{tau_j} rho

    where d is the dimension of the simplex, `omega_i` is the star of node `i`, `tau_j`
    a triangle in the star, and `b_j` the barycenter of the respective triangle.

    This method aims to preserve the mesh density and hence assumes `rho = 1/|tau|`.
    Incidentally, this makes the integral in the above formula vanish such that the
    energy gradient is a _linear_ function of the mesh coordinates. The linear operator
    is in fact given by the mesh graph Laplacian. Hence, one only needs to solve `d`
    systems with the graph Laplacian (one for each component).

    After this, one does edge-flipping and repeats the solve.
    """

    dim = 2

    def get_new_points(mesh, tol=1.0e-10):
        # Create matrix in IJV format
        cells = mesh.cells["nodes"].T

        row_idx = []
        col_idx = []
        val = []
        a = numpy.full(cells.shape[1], 2 / (dim + 1) ** 2)
        for i in [[0, 1], [1, 2], [2, 0]]:
            edges = cells[i]
            row_idx += [edges[0], edges[1], edges[0], edges[1]]
            col_idx += [edges[0], edges[1], edges[1], edges[0]]
            val += [+a, +a, -a, -a]

        row_idx = numpy.concatenate(row_idx)
        col_idx = numpy.concatenate(col_idx)
        val = numpy.concatenate(val)

        n = mesh.node_coords.shape[0]

        # Set Dirichlet conditions on the boundary
        matrix = scipy.sparse.coo_matrix((val, (row_idx, col_idx)), shape=(n, n))
        # Transform to CSR format for efficiency
        matrix = matrix.tocsr()

        # Apply Dirichlet conditions.
        verts = numpy.where(mesh.is_boundary_node)[0]
        # Set all Dirichlet rows to 0.
        for i in verts:
            matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]] = 0.0
        # Set the diagonal and RHS.
        d = matrix.diagonal()
        d[mesh.is_boundary_node] = 1.0
        matrix.setdiag(d)

        rhs = numpy.zeros((n, 2))
        rhs[mesh.is_boundary_node] = mesh.node_coords[mesh.is_boundary_node]

        # out = scipy.sparse.linalg.spsolve(matrix, rhs)
        ml = pyamg.ruge_stuben_solver(matrix)
        # Keep an eye on multiple rhs-solves in pyamg,
        # <https://github.com/pyamg/pyamg/issues/215>.
        out = numpy.column_stack(
            [ml.solve(rhs[:, 0], tol=tol), ml.solve(rhs[:, 1], tol=tol)]
        )
        return out[mesh.is_interior_node]

    return runner(get_new_points, *args, **kwargs)


def quasi_newton(*args, **kwargs):
    """Like linear_solve above, but assuming rho==1. Note that the energy gradient

        \\partial E_i = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) \\int_{tau_j} rho

    becomes

        \\partial E_i = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) |tau_j|.

    Because of the dependence of |tau_j| on the point coordinates, this is a nonlinear
    problem.

    This method makes the simplifying assumption that |tau_j| does in fact _not_ depend
    on the point coordinates. With this, one still only needs to solve a linear system.
    """

    dim = 2

    # def approx_hess_solve(x, grad):
    #     # TODO create mesh from x

    #     # Create matrix in IJV format
    #     cells = mesh.cells["nodes"].T
    #     row_idx = []
    #     col_idx = []
    #     val = []
    #     a = mesh.cell_volumes * (2 / (dim + 1) ** 2)
    #     for i in [[0, 1], [1, 2], [2, 0]]:
    #         edges = cells[i]
    #         row_idx += [edges[0], edges[1], edges[0], edges[1]]
    #         col_idx += [edges[0], edges[1], edges[1], edges[0]]
    #         val += [+a, +a, -a, -a]

    #     row_idx = numpy.concatenate(row_idx)
    #     col_idx = numpy.concatenate(col_idx)
    #     val = numpy.concatenate(val)

    #     n = mesh.node_coords.shape[0]

    #     # Set Dirichlet conditions on the boundary
    #     matrix = scipy.sparse.coo_matrix((val, (row_idx, col_idx)), shape=(n, n))
    #     # Transform to CSR format for efficiency
    #     matrix = matrix.tocsr()

    #     # Apply Dirichlet conditions.
    #     verts = numpy.where(mesh.is_boundary_node)[0]
    #     # Set all Dirichlet rows to 0.
    #     for i in verts:
    #         matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]] = 0.0
    #     # Set the diagonal and RHS.
    #     d = matrix.diagonal()
    #     d[mesh.is_boundary_node] = 1.0
    #     matrix.setdiag(d)

    #     rhs = numpy.zeros((n, 2))
    #     rhs[mesh.is_boundary_node] = mesh.node_coords[mesh.is_boundary_node]

    #     # out = scipy.sparse.linalg.spsolve(matrix, rhs)
    #     ml = pyamg.ruge_stuben_solver(matrix)
    #     # Keep an eye on multiple rhs-solves in pyamg,
    #     # <https://github.com/pyamg/pyamg/issues/215>.
    #     tol = 1.0e-10
    #     out = numpy.column_stack(
    #         [ml.solve(rhs[:, 0], tol=tol), ml.solve(rhs[:, 1], tol=tol)]
    #     )
    #     return out[mesh.is_interior_node]

    def get_new_points(mesh, tol=1.0e-10):
        return optipy.minimize(
            fun=fun,
            x0=[-1.0, 3.5],
            jac=jac,
            get_search_direction=approx_hess_solve,
            atol=tol
        )
    return runner(get_new_points, *args, **kwargs)

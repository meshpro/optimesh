"""Like linear_solve, but assuming rho==1. Note that the energy gradient

    \\partial E_i = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) \\int_{tau_j} rho

becomes

    \\partial E_i = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) |tau_j|.

Because of the dependence of |tau_j| on the point coordinates, this is a nonlinear
problem.

This method makes the simplifying assumption that |tau_j| does in fact _not_ depend
on the point coordinates. With this, one still only needs to solve a linear system.
"""
import meshplex
import numpy as np
import scipy.sparse.linalg
from numpy.typing import ArrayLike


def get_new_points(mesh: meshplex.Mesh) -> np.ndarray:
    # do one Newton step
    cells = mesh.cells("points")
    jac_x = _jac_uniform(mesh.points, cells)
    return mesh.points - _solve_hessian_approx_uniform(mesh.points, cells, jac_x)


def _jac_uniform(X: ArrayLike, cells: ArrayLike):
    """The approximated Jacobian is

      partial_i E = 2/(d+1) (x_i int_{omega_i} rho(x) dx - int_{omega_i} x rho(x) dx)
                  = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_{j, rho}) int_{tau_j} rho,

    see Chen-Holst. This method here assumes uniform density, rho(x) = 1, such that

      partial_i E = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) |tau_j|

    with b_j being the ordinary barycenter.
    """
    dim = 2
    mesh = meshplex.MeshTri(X, cells)

    X = np.asarray(X)
    jac = np.zeros(X.shape)
    for k in range(mesh.cells("points").shape[1]):
        i = mesh.cells("points")[:, k]
        vals = (mesh.points[i] - mesh.cell_barycenters).T * mesh.cell_volumes
        # np.add.at(jac, i, vals)
        jac += np.array([np.bincount(i, val, minlength=jac.shape[0]) for val in vals]).T

    return 2 / (dim + 1) * jac


def _solve_hessian_approx_uniform(
    X: ArrayLike, cells: ArrayLike, rhs: ArrayLike
) -> np.ndarray:
    """As discussed above, the approximated Jacobian is

      partial_i E = 2/(d+1) sum_{tau_j in omega_i} (x_i - b_j) |tau_j|.

    To get the Hessian, we have to form its derivative. As a simplifications,
    let us assume again that |tau_j| is independent of the point positions. Then we get

       partial_ii E = 2/(d+1) |omega_i| - 2/(d+1)**2 |omega_i|,
       partial_ij E = -2/(d+1)**2 |tau_j|.

    The terms with (d+1)**2 are from the barycenter in `partial_i E`. It turns out from
    numerical experiments that the negative term in `partial_ii E` is detrimental to the
    convergence. Hence, this approximated Hessian solver only considers the off-diagonal
    contributions from the barycentric terms.
    """
    dim = 2
    mesh = meshplex.MeshTri(X, cells)

    # Create matrix in IJV format
    row_idx = []
    col_idx = []
    val = []

    cells = mesh.cells("points").T
    X = np.asarray(X)
    n = X.shape[0]

    # Main diagonal, 2/(d+1) |omega_i| x_i
    a = mesh.cell_volumes * (2 / (dim + 1))
    for i in [0, 1, 2]:
        row_idx += [cells[i]]
        col_idx += [cells[i]]
        val += [a]

    # terms corresponding to -2/(d+1) * b_j |tau_j|
    a = mesh.cell_volumes * (2 / (dim + 1) ** 2)
    for i in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        edges = cells[i]
        # Leads to funny osciilatory movements
        # row_idx += [edges[0], edges[0], edges[0]]
        # col_idx += [edges[0], edges[1], edges[2]]
        # val += [-a, -a, -a]
        # Best so far
        row_idx += [edges[0], edges[0]]
        col_idx += [edges[1], edges[2]]
        val += [-a, -a]

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    val = np.concatenate(val)

    # Set Dirichlet conditions on the boundary
    matrix = scipy.sparse.coo_matrix((val, (row_idx, col_idx)), shape=(n, n))
    # Transform to CSR format for efficiency
    matrix = matrix.tocsr()

    # Apply Dirichlet conditions.
    # Set all Dirichlet rows to 0.
    for i in np.where(mesh.is_boundary_point)[0]:
        matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]] = 0.0
    # Set the diagonal and RHS.
    d = matrix.diagonal()
    d[mesh.is_boundary_point] = 1.0
    matrix.setdiag(d)

    rhs = np.asarray(rhs)
    rhs[mesh.is_boundary_point] = 0.0

    out = scipy.sparse.linalg.spsolve(matrix, rhs)

    # PyAMG fails on circleci.
    # ml = pyamg.ruge_stuben_solver(matrix)
    # # Keep an eye on multiple rhs-solves in pyamg,
    # # <https://github.com/pyamg/pyamg/issues/215>.
    # tol = 1.0e-10
    # out = np.column_stack(
    #     [ml.solve(rhs[:, 0], tol=tol), ml.solve(rhs[:, 1], tol=tol)]
    # )
    return out

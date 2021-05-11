import numpy as np

from ._helpers import jac_uniform


def get_new_points(mesh):
    """Lloyd's algorithm can be though of a diagonal-only Hessian; this method
    incorporates the diagonal blocks, too. It's almost as cheap but performs better.
    """
    # Exclude all cells which have a too negative covolume-edgelength ratio. This is
    # necessary to prevent points to be dragged outside of the domain by very flat cells
    # on the boundary.
    # There are other possible heuristics too. For example, one could restrict the mask
    # to cells at or near the boundary.
    mask = np.any(mesh.ce_ratios < -0.5, axis=0)

    X = mesh.points.copy()
    # Collect the diagonal blocks.
    diagonal_blocks = np.zeros((X.shape[0], X.shape[1], X.shape[1]))

    # First the Lloyd part.
    cv = mesh.get_control_volumes(cell_mask=mask)
    for k in range(X.shape[1]):
        diagonal_blocks[:, k, k] += 2 * cv

    hec = mesh.half_edge_coords[:, ~mask]
    ei_outer_ei = np.einsum("ijk, ijl->ijkl", hec, hec)

    # Without adding the control volumes above, the update would be
    # ```
    # m = 0.5 * ce * (np.eye(X.shape[1]) * np.dot(ei, ei) - ei_outer_ei)
    # ```
    # instead of
    # ```
    # m = -0.5 * ce * ei_outer_ei
    # ```
    # This makes clear why the diagonal blocks are always positive definite if the
    # mesh is Delaunay.
    M = -0.5 * ei_outer_ei * mesh.ce_ratios[:, ~mask, None, None]

    # dg = diagonal_blocks.copy()
    # np.add.at(dg, mesh.idx[-1][0][:, ~mask], M)
    # np.add.at(dg, mesh.idx[-1][1][:, ~mask], M)

    n = diagonal_blocks.shape[0]
    diagonal_blocks += np.array(
        [
            [
                np.bincount(
                    mesh.idx[-1][0][:, ~mask].reshape(-1),
                    M[..., i, j].reshape(-1),
                    minlength=n,
                )
                for j in range(diagonal_blocks.shape[2])
            ]
            for i in range(diagonal_blocks.shape[1])
        ]
    ).T
    diagonal_blocks += np.array(
        [
            [
                np.bincount(
                    mesh.idx[-1][1][:, ~mask].reshape(-1),
                    M[..., i, j].reshape(-1),
                    minlength=n,
                )
                for j in range(diagonal_blocks.shape[2])
            ]
            for i in range(diagonal_blocks.shape[1])
        ]
    ).T

    rhs = -jac_uniform(mesh, mask)

    # When using a cell mask, it can happen that some points don't get any contribution
    # at all because they are adjacent only to masked cells.
    idx = np.any(np.isnan(rhs), axis=1)
    diagonal_blocks[idx] = 0.0
    for k in range(X.shape[1]):
        diagonal_blocks[idx, k, k] = 1.0
    rhs[idx] = 0.0

    X += np.linalg.solve(diagonal_blocks, rhs)

    return X

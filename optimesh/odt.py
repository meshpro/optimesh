"""
Optimal Delaunay Tesselation.

Long Chen, Michael Holst,
Efficient mesh optimization schemes based on Optimal Delaunay Triangulations,
Comput. Methods Appl. Mech. Engrg. 200 (2011) 967–984,
<https://doi.org/10.1016/j.cma.2010.11.007>.
"""
import numpy

import quadpy
from meshplex import MeshTri

from .helpers import (
    get_new_points_count_averaged,
    get_new_points_volume_averaged,
    print_stats,
    runner,
)


def energy(mesh, uniform_density=False):
    """The mesh energy is defined as

    E = int_Omega |u_l(x) - u(x)| rho(x) dx

    where u(x) = ||x||^2 and u_l is its piecewise linearization on the mesh.
    """
    # E = 1/(d+1) sum_i ||x_i||^2 |omega_i| - int_Omega_i ||x||^2
    dim = mesh.cells["nodes"].shape[1] - 1

    n = mesh.node_coords.shape[0]
    star_volume = numpy.zeros(n)
    for i in range(3):
        idx = mesh.cells["nodes"][:, i]
        if uniform_density:
            # rho = 1,
            # int_{star} phi_i * rho = 1/(d+1) sum_{triangles in star} |triangle|
            # numpy.add.at(star_volume, idx, mesh.cell_volumes)
            star_volume += numpy.bincount(idx, mesh.cell_volumes, minlength=n)
        else:
            # rho = 1 / tau_j,
            # int_{star} phi_i * rho = 1/(d+1) |num triangles in star|
            # numpy.add.at(star_volume, idx, numpy.ones(idx.shape, dtype=float))
            star_volume += numpy.bincount(idx, numpy.ones(idx.shape), minlength=n)
    x2 = numpy.einsum("ij,ij->i", mesh.node_coords, mesh.node_coords)
    out = 1 / (dim + 1) * numpy.dot(star_volume, x2)

    # could be cached
    assert dim == 2
    x = mesh.node_coords[:, :2]
    triangles = numpy.moveaxis(x[mesh.cells["nodes"]], 0, 1)
    # Take any scheme with order 2
    scheme = quadpy.triangle.dunavant_02()
    val = scheme.integrate(lambda x: x[0] ** 2 + x[1] ** 2, triangles)
    if uniform_density:
        val = numpy.sum(val)
    else:
        rho = 1.0 / mesh.cell_volumes
        val = numpy.dot(val, rho)

    assert out >= val

    return out - val


def fixed_point_uniform(points, cells, *args, **kwargs):
    """Idea:
    Move interior mesh points into the weighted averages of the circumcenters
    of their adjacent cells. If a triangle cell switches orientation in the
    process, don't move quite so far.
    """

    def get_new_points(mesh):
        # Get circumcenters everywhere except at cells adjacent to the boundary;
        # barycenters there.
        cc = mesh.cell_circumcenters
        bc = mesh.cell_barycenters
        # Find all cells with a boundary edge
        boundary_cell_ids = mesh.edges_cells[1][:, 0]
        cc[boundary_cell_ids] = bc[boundary_cell_ids]
        return get_new_points_volume_averaged(mesh, cc)

    mesh = MeshTri(points, cells)
    runner(
        get_new_points,
        mesh,
        *args,
        **kwargs,
        method_name="Optimal Delaunay Tesselation (ODT), uniform density, fixed-point variant",
    )
    return mesh.node_coords, mesh.cells["nodes"]


def fixed_point_density_preserving(points, cells, *args, **kwargs):
    """Idea:
    Move interior mesh points into the weighted averages of the circumcenters
    of their adjacent cells.
    """

    def get_new_points(mesh):
        # Get circumcenters everywhere except at cells adjacent to the boundary;
        # barycenters there. The reason is that points near the boundary would be
        # "sucked" out of the domain if the boundary cell is very flat, i.e., its
        # circumcenter is very far outside of the domain.
        # This heuristic also applies to cells _near_ the boundary, though, and if
        # constructed maliciously, any mesh. Hence, this method can break down. A better
        # approach is to use barycenters for all cells which are sufficiently flat.
        cc = mesh.cell_circumcenters.copy()
        # Find all cells with a boundary edge
        is_boundary_cell = (
            numpy.sum(mesh.is_boundary_node[mesh.cells["nodes"]], axis=1) == 2
        )
        cc[is_boundary_cell] = mesh.cell_barycenters[is_boundary_cell]
        return get_new_points_count_averaged(mesh, cc)

    mesh = MeshTri(points, cells)
    runner(
        get_new_points,
        mesh,
        *args,
        **kwargs,
        method_name="Optimal Delaunay Tesselation (ODT), density-preserving, fixed-point variant",
    )
    return mesh.node_coords, mesh.cells["nodes"]


def nonlinear_optimization_uniform(
    X,
    cells,
    tol,
    max_num_steps,
    verbose=False,
    step_filename_format=None,
    callback=None,
):
    """Optimal Delaunay Tesselation smoothing.

    This method minimizes the energy

        E = int_Omega |u_l(x) - u(x)| rho(x) dx

    where u(x) = ||x||^2, u_l is its piecewise linear nodal interpolation and
    rho is the density. Since u(x) is convex, u_l >= u everywhere and

        u_l(x) = sum_i phi_i(x) u(x_i)

    where phi_i is the hat function at x_i. With rho(x)=1, this gives

        E = int_Omega sum_i phi_i(x) u(x_i) - u(x)
          = 1/(d+1) sum_i ||x_i||^2 |omega_i| - int_Omega ||x||^2

    where d is the spatial dimension and omega_i is the star of x_i (the set of
    all simplices containing x_i).
    """
    import scipy.optimize

    mesh = MeshTri(X, cells)

    if step_filename_format:
        mesh.save(
            step_filename_format.format(0),
            show_coedges=False,
            show_axes=False,
            cell_quality_coloring=("viridis", 0.0, 1.0, False),
        )

    if verbose:
        print("Before:")
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)

    def f(x):
        mesh.node_coords[mesh.is_interior_node] = x.reshape(-1, X.shape[1])
        mesh.update_values()
        return energy(mesh, uniform_density=True)

    # TODO put f and jac together
    def jac(x):
        mesh.node_coords[mesh.is_interior_node] = x.reshape(-1, X.shape[1])
        mesh.update_values()

        grad = numpy.zeros(mesh.node_coords.shape)
        n = grad.shape[0]
        cc = mesh.cell_circumcenters
        for mcn in mesh.cells["nodes"].T:
            vals = (mesh.node_coords[mcn] - cc).T * mesh.cell_volumes
            # numpy.add.at(grad, mcn, vals)
            grad += numpy.array(
                [numpy.bincount(mcn, val, minlength=n) for val in vals]
            ).T
        gdim = 2
        grad *= 2 / (gdim + 1)
        return grad[mesh.is_interior_node].flatten()

    def flip_delaunay(x):
        flip_delaunay.step += 1
        # Flip the edges
        mesh.node_coords[mesh.is_interior_node] = x.reshape(-1, X.shape[1])
        mesh.update_values()
        mesh.flip_until_delaunay()

        if step_filename_format:
            mesh.save(
                step_filename_format.format(flip_delaunay.step),
                show_coedges=False,
                show_axes=False,
                cell_quality_coloring=("viridis", 0.0, 1.0, False),
            )

        if callback:
            callback(flip_delaunay.step, mesh)

        # mesh.show()
        # exit(1)
        return

    flip_delaunay.step = 0

    x0 = X[mesh.is_interior_node].flatten()

    if callback:
        callback(0, mesh)

    out = scipy.optimize.minimize(
        f,
        x0,
        jac=jac,
        # method="Nelder-Mead",
        # method="Powell",
        # method="CG",
        # method="Newton-CG",
        method="BFGS",
        # method="L-BFGS-B",
        # method="TNC",
        # method="COBYLA",
        # method="SLSQP",
        tol=tol,
        callback=flip_delaunay,
        options={"maxiter": max_num_steps},
    )
    # Don't assert out.success; max_num_steps may be reached, that's fine.

    # One last edge flip
    mesh.node_coords[mesh.is_interior_node] = out.x.reshape(-1, X.shape[1])
    mesh.update_values()
    mesh.flip_until_delaunay()

    info = (
        "{} steps,".format(out.nit)
        + "Optimal Delaunay Tesselation (ODT), uniform density, BFGS variant"
    )
    if verbose:
        print("\nFinal ({})".format(info))
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)
        print()

    return mesh.node_coords, mesh.cells["nodes"]

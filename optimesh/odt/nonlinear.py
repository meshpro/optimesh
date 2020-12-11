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

from ..helpers import print_stats


def _energy(mesh, uniform_density=False):
    """The mesh energy is defined as

    E = int_Omega |u_l(x) - u(x)| rho(x) dx

    where u(x) = ||x||^2 and u_l is its piecewise linearization on the mesh.
    """
    # E = 1/(d+1) sum_i ||x_i||^2 |omega_i| - int_Omega_i ||x||^2
    dim = mesh.cells["points"].shape[1] - 1

    n = mesh.points.shape[0]
    star_volume = numpy.zeros(n)
    for i in range(3):
        idx = mesh.cells["points"][:, i]
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
    x2 = numpy.einsum("ij,ij->i", mesh.points, mesh.points)
    out = 1 / (dim + 1) * numpy.dot(star_volume, x2)

    # could be cached
    assert dim == 2
    x = mesh.points[:, :2]
    triangles = numpy.moveaxis(x[mesh.cells["points"]], 0, 1)
    # Get a scheme with order 2
    scheme = quadpy.t2.get_good_scheme(2)
    val = scheme.integrate(lambda x: x[0] ** 2 + x[1] ** 2, triangles)
    if uniform_density:
        val = numpy.sum(val)
    else:
        rho = 1.0 / mesh.cell_volumes
        val = numpy.dot(val, rho)

    assert out >= val
    return out - val


def nonlinear_optimization(
    mesh,
    method,
    # method="BFGS",
    # method="Nelder-Mead",
    # method="Powell",
    # method="CG",
    # method="Newton-CG",
    # method="L-BFGS-B",
    # method="TNC",
    # method="COBYLA",
    # method="SLSQP",
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

    X = mesh.points

    if step_filename_format:
        mesh.save(
            step_filename_format.format(0),
            show_coedges=False,
            show_axes=False,
            cell_quality_coloring=("viridis", 0.0, 1.0, False),
        )

    if verbose:
        print("Before:")
        extra_cols = ["energy: {:.5e}".format(_energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)

    def f(x):
        mesh.set_points(x.reshape(-1, X.shape[1]), mesh.is_interior_point)
        return _energy(mesh, uniform_density=True)

    # TODO put f and jac together
    def jac(x):
        mesh.set_points(x.reshape(-1, X.shape[1]), mesh.is_interior_point)

        grad = numpy.zeros(mesh.points.shape)
        n = grad.shape[0]
        cc = mesh.cell_circumcenters
        for mcn in mesh.cells["points"].T:
            vals = (mesh.points[mcn] - cc).T * mesh.cell_volumes
            # numpy.add.at(grad, mcn, vals)
            grad += numpy.array(
                [numpy.bincount(mcn, val, minlength=n) for val in vals]
            ).T
        gdim = 2
        grad *= 2 / (gdim + 1)
        return grad[mesh.is_interior_point].flatten()

    def flip_delaunay(x):
        flip_delaunay.step += 1
        # Flip the edges
        mesh.set_points(x.reshape(-1, X.shape[1]), mesh.is_interior_point)
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

    flip_delaunay.step = 0

    x0 = X[mesh.is_interior_point].flatten()

    if callback:
        callback(0, mesh)

    out = scipy.optimize.minimize(
        f,
        x0,
        jac=jac,
        method=method,
        tol=tol,
        callback=flip_delaunay,
        options={"maxiter": max_num_steps},
    )
    # Don't assert out.success; max_num_steps may be reached, that's fine.

    # One last edge flip
    mesh.set_points(out.x.reshape(-1, X.shape[1]), mesh.is_interior_point)

    mesh.flip_until_delaunay()

    info = (
        f"{out.nit} steps,"
        + "Optimal Delaunay Tesselation (ODT), uniform density, BFGS variant"
    )
    if verbose:
        print(f"\nFinal ({info})")
        extra_cols = ["energy: {:.5e}".format(_energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)
        print()

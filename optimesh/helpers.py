# -*- coding: utf-8 -*-
#
import numpy
import fastfunc
from meshplex import MeshTri
import quadpy

import asciiplotlib as apl


def print_stats(mesh, extra_cols=None):
    extra_cols = [] if extra_cols is None else extra_cols

    angles = mesh.get_angles() / numpy.pi * 180
    angles_hist, angles_bin_edges = numpy.histogram(
        angles, bins=numpy.linspace(0.0, 180.0, num=73, endpoint=True)
    )

    q = mesh.get_quality()
    q_hist, q_bin_edges = numpy.histogram(
        q, bins=numpy.linspace(0.0, 1.0, num=41, endpoint=True)
    )

    grid = apl.subplot_grid(
        (1, 4 + len(extra_cols)), column_widths=None, border_style=None
    )
    grid[0, 0].hist(angles_hist, angles_bin_edges, grid=[24], bar_width=1, strip=True)
    grid[0, 1].aprint("min angle:     {:7.3f}".format(numpy.min(angles)))
    grid[0, 1].aprint("avg angle:     {:7.3f}".format(60))
    grid[0, 1].aprint("max angle:     {:7.3f}".format(numpy.max(angles)))
    grid[0, 1].aprint("std dev angle: {:7.3f}".format(numpy.std(angles)))
    grid[0, 2].hist(q_hist, q_bin_edges, bar_width=1, strip=True)
    grid[0, 3].aprint("min quality: {:5.3f}".format(numpy.min(q)))
    grid[0, 3].aprint("avg quality: {:5.3f}".format(numpy.average(q)))
    grid[0, 3].aprint("max quality: {:5.3f}".format(numpy.max(q)))

    for k, col in enumerate(extra_cols):
        grid[0, 4 + k].aprint(col)

    grid.show()
    return


def write(mesh, file_prefix, filetype, k):
    if filetype == "png":
        from matplotlib import pyplot as plt

        fig = mesh.plot(show_coedges=False, show_centroids=False, show_axes=False)
        fig.suptitle("step {}".format(k), fontsize=20)
        plt.savefig("{}{:04d}.png".format(file_prefix, k))
        plt.close(fig)
        return

    mesh.write("{}{:04d}.{}".format(file_prefix, k, filetype))
    return


def sit_in_plane(X, tol=1.0e-15):
    """Checks if all points X sit in a plane.
    """
    orth = numpy.cross(X[1] - X[0], X[2] - X[0])
    orth /= numpy.sqrt(numpy.dot(orth, orth))
    return (abs(numpy.dot(X - X[0], orth)) < tol).all()


def energy(mesh, uniform_density=False):
    """The mesh energy is defined as

    E = int_Omega |u_l(x) - u(x)| rho(x) dx

    where u(x) = ||x||^2 and u_l is its piecewise linearization on the mesh.
    """
    # E = 1/(d+1) sum_i ||x_i||^2 |omega_i| - int_Omega_i ||x||^2
    dim = mesh.cells["nodes"].shape[1] - 1

    star_volume = numpy.zeros(mesh.node_coords.shape[0])
    for i in range(3):
        idx = mesh.cells["nodes"][:, i]
        if uniform_density:
            # rho = 1,
            # int_{star} phi_i * rho = 1/(d+1) sum_{triangles in star} |triangle|
            fastfunc.add.at(star_volume, idx, mesh.cell_volumes)
        else:
            # rho = 1 / tau_j,
            # int_{star} phi_i * rho = 1/(d+1) |num triangles in star|
            fastfunc.add.at(star_volume, idx, numpy.ones(idx.shape, dtype=float))
    x2 = numpy.einsum("ij,ij->i", mesh.node_coords, mesh.node_coords)
    out = 1 / (dim + 1) * numpy.dot(star_volume, x2)

    # could be cached
    assert dim == 2
    x = mesh.node_coords[:, :2]
    triangles = numpy.moveaxis(x[mesh.cells["nodes"]], 0, 1)
    val = quadpy.triangle.integrate(
        lambda x: x[0] ** 2 + x[1] ** 2,
        triangles,
        # Take any scheme with order 2
        quadpy.triangle.Dunavant(2),
    )
    if uniform_density:
        val = numpy.sum(val)
    else:
        rho = 1.0 / mesh.cell_volumes
        val = numpy.dot(val, rho)

    assert out >= val

    return out - val


def runner(
    get_new_interior_points,
    X,
    cells,
    tol,
    max_num_steps,
    verbosity=1,
    step_filename_format=None,
    uniform_density=False,
):
    if X.shape[1] == 3:
        # create flat mesh
        assert numpy.all(abs(X[:, 2]) < 1.0e-15)
        X = X[:, :2]

    mesh = MeshTri(X, cells, flat_cell_correction=None)
    mesh.flip_until_delaunay()

    if step_filename_format:
        mesh.save(
            step_filename_format.format(0),
            show_centroids=False,
            show_coedges=False,
            show_axes=False,
            nondelaunay_edge_color="k",
        )

    if verbosity > 0:
        print("Before:")
        extra_cols = [
            "energy: {:.5e}".format(energy(mesh, uniform_density=uniform_density))
        ]
        print_stats(mesh, extra_cols=extra_cols)

    mesh.mark_boundary()

    k = 0
    while True:
        k += 1

        new_interior_points = get_new_interior_points(mesh)

        original_orient = mesh.get_signed_tri_areas() > 0.0
        original_coords = mesh.node_coords[mesh.is_interior_node]

        # Step unless the orientation of any cell changes.
        alpha = 1.0
        while True:
            xnew = (1 - alpha) * original_coords + alpha * new_interior_points
            mesh.update_interior_node_coordinates(xnew)
            new_orient = mesh.get_signed_tri_areas() > 0.0
            if numpy.all(original_orient == new_orient):
                break
            alpha /= 2

        mesh.flip_until_delaunay()

        if step_filename_format:
            mesh.save(
                step_filename_format.format(k),
                show_centroids=False,
                show_coedges=False,
                show_axes=False,
                nondelaunay_edge_color="k",
            )

        # Abort the loop if the update is small
        diff = mesh.node_coords[mesh.is_interior_node] - original_coords
        if numpy.all(numpy.einsum("ij,ij->i", diff, diff) < tol ** 2):
            break

        if k >= max_num_steps:
            break

        if verbosity > 1:
            print("\nstep {}:".format(k))
            print_stats(mesh)

    if verbosity > 0:
        print("\nFinal ({} steps):".format(k))
        extra_cols = [
            "energy: {:.5e}".format(energy(mesh, uniform_density=uniform_density))
        ]
        print_stats(mesh, extra_cols=extra_cols)
        print()

    return mesh.node_coords, mesh.cells["nodes"]

# -*- coding: utf-8 -*-
#
import numpy
import fastfunc
from meshplex import MeshTri

import asciiplotlib as apl


def print_stats(mesh, extra_cols=None):
    extra_cols = [] if extra_cols is None else extra_cols

    angles = mesh.angles / numpy.pi * 180
    angles_hist, angles_bin_edges = numpy.histogram(
        angles, bins=numpy.linspace(0.0, 180.0, num=73, endpoint=True)
    )

    q = mesh.triangle_quality
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


def sit_in_plane(X, tol=1.0e-15):
    """Checks if all points X sit in a plane.
    """
    orth = numpy.cross(X[1] - X[0], X[2] - X[0])
    orth /= numpy.sqrt(numpy.dot(orth, orth))
    return (abs(numpy.dot(X - X[0], orth)) < tol).all()


def runner(
    get_new_interior_points,
    X,
    cells,
    tol,
    max_num_steps,
    verbosity=1,
    callback=None,
    step_filename_format=None,
    uniform_density=False,
    update_coordinates=lambda mesh, xnew: mesh.update_interior_node_coordinates(xnew),
):
    if X.shape[1] == 3:
        # create flat mesh
        assert numpy.all(abs(X[:, 2]) < 1.0e-15)
        X = X[:, :2]

    mesh = MeshTri(X, cells)
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
        print_stats(mesh)

    k = 0
    if callback:
        callback(k, mesh)

    while True:
        k += 1

        new_interior_points = get_new_interior_points(mesh)

        original_orient = mesh.signed_cell_areas > 0.0
        original_interior_coords = mesh.node_coords[mesh.is_interior_node]

        # Step unless the orientation of any cell changes.
        alpha = 1.0
        while True:
            xnew = (1 - alpha) * original_interior_coords + alpha * new_interior_points
            update_coordinates(mesh, xnew)
            new_orient = mesh.signed_cell_areas > 0.0
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

        if callback:
            callback(k, mesh)

        # Abort the loop if the update is small
        diff = mesh.node_coords[mesh.is_interior_node] - original_interior_coords
        if numpy.all(numpy.einsum("ij,ij->i", diff, diff) < tol ** 2):
            break

        if k >= max_num_steps:
            break

        if verbosity > 1:
            print("\nstep {}:".format(k))
            print_stats(mesh)

    if verbosity > 0:
        print("\nFinal ({} steps):".format(k))
        print_stats(mesh)
        print()

    return mesh.node_coords, mesh.cells["nodes"]


def get_new_points_volume_averaged(mesh, reference_points):
    scaled_rp = (reference_points.T * mesh.cell_volumes).T

    weighted_rp_average = numpy.zeros(mesh.node_coords.shape)
    for i in mesh.cells["nodes"].T:
        fastfunc.add.at(weighted_rp_average, i, scaled_rp)

    omega = numpy.zeros(len(mesh.node_coords))
    for i in mesh.cells["nodes"].T:
        fastfunc.add.at(omega, i, mesh.cell_volumes)

    idx = mesh.is_interior_node
    new_points = (weighted_rp_average[idx].T / omega[idx]).T
    return new_points


def get_new_points_count_averaged(mesh, reference_points):
    # Estimate the density as 1/|tau|. This leads to some simplifcations: The
    # new point is simply the average of of the reference points
    # (barycenters/cirumcenters) in the star.
    rp_average = numpy.zeros(mesh.node_coords.shape)
    for i in mesh.cells["nodes"].T:
        fastfunc.add.at(rp_average, i, reference_points)

    omega = numpy.zeros(len(mesh.node_coords))
    for i in mesh.cells["nodes"].T:
        fastfunc.add.at(omega, i, numpy.ones(i.shape, dtype=float))

    idx = mesh.is_interior_node
    new_points = (rp_average[idx].T / omega[idx]).T
    return new_points

# -*- coding: utf-8 -*-
#
import numpy
import fastfunc

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


def runner(
    get_new_interior_points,
    mesh,
    tol,
    max_num_steps,
    verbose=False,
    callback=None,
    step_filename_format=None,
    uniform_density=False,
    straighten_out=lambda mesh: mesh.flip_until_delaunay(),
    get_stats_mesh=lambda mesh: mesh,
):
    k = 0

    stats_mesh = get_stats_mesh(mesh)
    print("\nBefore:")
    print_stats(stats_mesh)
    if step_filename_format:
        stats_mesh.save(
            step_filename_format.format(k),
            show_centroids=False,
            show_coedges=False,
            show_axes=False,
            nondelaunay_edge_color="k",
        )

    if callback:
        callback(k, mesh)

    straighten_out(mesh)

    while True:
        k += 1

        new_interior_points = get_new_interior_points(mesh)

        original_interior_coords = mesh.node_coords[mesh.is_interior_node]

        # We previously checked here if the orientation of any cell changes and
        # reduced the step size if it did. Computing the orientation is unnecessarily
        # costly though and doesn't easily translate to shell meshes. Since orientation
        # changes cannot occur, e.g., with CPT, advice the user to apply a few steps of
        # a robust smoother first (CPT) if the method crashes.
        mesh.node_coords[mesh.is_interior_node] = new_interior_points
        mesh.update_values()
        straighten_out(mesh)

        # Abort the loop if the update is small
        diff = new_interior_points - original_interior_coords
        is_final = (
            numpy.all(numpy.einsum("ij,ij->i", diff, diff) < tol ** 2)
            or k >= max_num_steps
        )

        stats_mesh = get_stats_mesh(mesh)
        if verbose and not is_final:
            print("\nstep {}:".format(k))
            print_stats(stats_mesh)
        elif is_final:
            print("\nFinal ({} steps):".format(k))
            print_stats(stats_mesh)
        if step_filename_format:
            stats_mesh.save(
                step_filename_format.format(k),
                show_centroids=False,
                show_coedges=False,
                show_axes=False,
                nondelaunay_edge_color="k",
            )
        if callback:
            callback(k, mesh)

        if is_final:
            break

    return


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

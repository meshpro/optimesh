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
import numpy
import fastfunc
from meshplex import MeshTri

from .helpers import print_stats, energy


def odt(X, cells, tol, max_num_steps, verbosity=1):
    """Optimal Delaunay Triangulation.

    Idea:
    Move interior mesh points into the weighted averages of the circumcenters
    of their adjacent cells. If a triangle cell switches orientation in the
    process, don't move quite so far.
    """
    return _run(
        lambda mesh: mesh.get_cell_circumcenters(),
        X,
        cells,
        tol,
        max_num_steps,
        verbosity=verbosity,
    )


def cpt(X, cells, tol, max_num_steps, verbosity=1):
    """Centroidal Patch Triangulation. Mimics the definition of Centroidal
    Voronoi Tessellations for which the generator and centroid of each Voronoi
    region coincide.

    Idea:
    Move interior mesh points into the weighted averages of the centroids
    (barycenters) of their adjacent cells. If a triangle cell switches
    orientation in the process, don't move quite so far.
    """
    return _run(
        lambda mesh: mesh.get_centroids(),
        X,
        cells,
        tol,
        max_num_steps,
        verbosity=verbosity,
    )


def _run(get_reference_points_, X, cells, tol, max_num_steps, verbosity=1):
    if X.shape[1] == 3:
        # create flat mesh
        assert numpy.all(abs(X[:, 2]) < 1.0e-15)
        X = X[:, :2]

    mesh = MeshTri(X, cells, flat_cell_correction=None)
    mesh.flip_until_delaunay()

    # mesh.save_png(
    #     'step{:03d}'.format(0), show_centroids=False, show_coedges=False
    #     )

    if verbosity > 0:
        print("Before:")
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)

    mesh.mark_boundary()

    k = 0
    while True:
        k += 1

        rp = get_reference_points_(mesh)
        scaled_rp = (rp.T * mesh.cell_volumes).T

        weighted_rp_average = numpy.zeros(mesh.node_coords.shape)
        for i in mesh.cells["nodes"].T:
            fastfunc.add.at(weighted_rp_average, i, scaled_rp)

        omega = numpy.zeros(len(mesh.node_coords))
        for i in mesh.cells["nodes"].T:
            fastfunc.add.at(omega, i, mesh.cell_volumes)

        weighted_rp_average = (weighted_rp_average.T / omega).T

        original_orient = mesh.get_signed_tri_areas() > 0.0
        original_coords = mesh.node_coords.copy()

        # Step unless the orientation of any cell changes.
        alpha = 1.0
        while True:
            xnew = (1 - alpha) * original_coords + alpha * weighted_rp_average
            # Preserve boundary nodes
            xnew[mesh.is_boundary_node] = original_coords[mesh.is_boundary_node]
            mesh.update_node_coordinates(xnew)
            new_orient = mesh.get_signed_tri_areas() > 0.0
            if numpy.all(original_orient == new_orient):
                break
            alpha /= 2

        mesh.flip_until_delaunay()

        # mesh.save_png(
        #     'step{:03d}'.format(k), show_centroids=False, show_coedges=False
        #     )

        if verbosity > 1:
            print("\nstep {}:".format(k + 1))
            print_stats(mesh)

        # Abort the loop if the update is small
        diff = mesh.node_coords - original_coords
        if numpy.all(numpy.einsum("ij,ij->i", diff, diff) < tol ** 2):
            break

        if k >= max_num_steps:
            break

    if verbosity > 0:
        print("\nFinal ({} steps):".format(k))
        extra_cols = ["energy: {:.5e}".format(energy(mesh))]
        print_stats(mesh, extra_cols=extra_cols)
        print()

    return mesh.node_coords, mesh.cells["nodes"]

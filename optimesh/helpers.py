# -*- coding: utf-8 -*-
#
import numpy
import fastfunc
import quadpy
from voropy.mesh_tri import MeshTri

import asciiplotlib as apl


def gather_stats(mesh):
    # The cosines of the angles are the negative dot products of
    # the normalized edges adjacent to the angle.
    norms = numpy.sqrt(mesh.ei_dot_ei)
    normalized_ei_dot_ej = numpy.array(
        [
            mesh.ei_dot_ej[0] / norms[1] / norms[2],
            mesh.ei_dot_ej[1] / norms[2] / norms[0],
            mesh.ei_dot_ej[2] / norms[0] / norms[1],
        ]
    )
    angles = numpy.arccos(-normalized_ei_dot_ej) / (2 * numpy.pi) * 360.0

    hist, bin_edges = numpy.histogram(
        angles, bins=numpy.linspace(0.0, 180.0, num=73, endpoint=True)
    )
    return hist, bin_edges, angles


def print_stats(hist, bin_edges, angles, extra_cols=None):
    extra_cols = [] if extra_cols is None else extra_cols

    n = len(extra_cols)

    grid = apl.subplot_grid((1, 2 + n), column_widths=None, border_style=None)
    grid[0, 0].hist(hist, bin_edges, grid=[24], bar_width=1, strip=True)
    grid[0, 1].aprint("min angle:     {:7.3f}".format(numpy.min(angles)))
    grid[0, 1].aprint("av angle:      {:7.3f}".format(60))
    grid[0, 1].aprint("max angle:     {:7.3f}".format(numpy.max(angles)))
    grid[0, 1].aprint("std dev angle: {:7.3f}".format(numpy.std(angles)))

    for k, col in enumerate(extra_cols):
        grid[0, 2 + k].aprint(col)

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


def get_boundary_edge_ratio(X, cells):
    """Gets the ratio of the longest vs. the shortest boundary edge.
    """
    submesh = MeshTri(X, cells, flat_cell_correction="full")
    submesh.create_edges()
    x = submesh.node_coords[submesh.idx_hierarchy[..., submesh.is_boundary_edge]]
    e = x[0] - x[1]
    edge_lengths2 = numpy.einsum("ij, ij->i", e, e)
    return numpy.sqrt(max(edge_lengths2) / min(edge_lengths2))


def extract_submesh_entities(X, cells, cell_in_submesh):
    # Get cells
    submesh_cells = cells[cell_in_submesh]
    # Get the vertices
    submesh_verts, uidx = numpy.unique(submesh_cells, return_inverse=True)
    submesh_X = X[submesh_verts]
    #
    submesh_cells = uidx.reshape(submesh_cells.shape)
    return submesh_X, submesh_cells, submesh_verts


def energy(mesh):
    """The mesh energy is defined as

    E = int_Omega |u_l(x) - u(x)| rho(x) dx

    where u(x) = ||x||^2 and u_l is its piecewise linearization on the mesh.
    """
    # E = 1/(d+1) sum_i ||x_i||^2 |omega_i| - int_Omega_i ||x||^2
    star_volume = numpy.zeros(mesh.node_coords.shape[0])

    dim = mesh.cells["nodes"].shape[1] - 1

    for i in range(3):
        fastfunc.add.at(star_volume, mesh.cells["nodes"][:, i], mesh.cell_volumes)
    x2 = numpy.einsum("ij,ij->i", mesh.node_coords, mesh.node_coords)
    out = 1 / (dim + 1) * numpy.dot(star_volume, x2)

    # could be cached
    assert dim == 2
    x = mesh.node_coords[:, :2]
    triangles = numpy.moveaxis(x[mesh.cells["nodes"]], 0, 1)
    val = quadpy.triangle.integrate(
        lambda x: x[0]**2 + x[1]**2, triangles,
        # Take any scheme with order 2
        quadpy.triangle.Dunavant(2)
    )
    val = numpy.sum(val)

    return out - val

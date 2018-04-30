# -*- coding: utf-8 -*-
#
from __future__ import print_function

from dolfin import (
    Mesh, MeshEditor, FunctionSpace, Expression, assemble, dx
    )
import numpy
from voropy.mesh_tri import MeshTri
import scipy.optimize


from .helpers import (
    sit_in_plane, gather_stats, write, flip_until_delaunay, print_stats
    )


def odt(X, cells, verbose=True, output_filetype=None):
    '''Perform k steps of Laplacian smoothing to the mesh, i.e., moving each
    interior vertex to the arithmetic average of its neighboring points.
    '''
    # flat mesh
    assert numpy.all(abs(X[:, 2]) < 1.0e-15)

    mesh = MeshTri(X, cells, flat_cell_correction=None)
    initial_stats = gather_stats(mesh)

    boundary_verts = mesh.get_boundary_vertices()

    is_interior_node = ~mesh.is_boundary_node

    # flat triangles
    gdim = 2

    def f(x):
        interior_coords = x.reshape(-1, 2)
        interior_coords = numpy.column_stack([
            interior_coords, numpy.zeros(len(interior_coords))
            ])
        coords = X.copy()
        coords[is_interior_node] = interior_coords

        voropy_mesh = MeshTri(coords, cells, flat_cell_correction=None)
        # voropy_mesh.show()
        voropy_mesh, _ = flip_until_delaunay(voropy_mesh)

        # if verbose:
        #     print('\nstep: {}'.format(k))
        #     print_stats([gather_stats(voropy_mesh)])

        # create dolfin mesh
        editor = MeshEditor()
        dolfin_mesh = Mesh()
        # topological and geometrical dimension 2
        editor.open(dolfin_mesh, 'triangle', 2, 2, 1)
        editor.init_vertices(len(voropy_mesh.node_coords))
        editor.init_cells(len(cells))
        for k, point in enumerate(voropy_mesh.node_coords):
            editor.add_vertex(k, point[:2])
        for k, cell in enumerate(voropy_mesh.cells['nodes'].astype(numpy.uintp)):
            editor.add_cell(k, cell)
        editor.close()

        V = FunctionSpace(dolfin_mesh, 'CG', 1)
        q = Expression('x[0]*x[0] + x[1]*x[1]', element = V.ufl_element())
        out = assemble(q * dx(dolfin_mesh))
        # print(out)
        return out

    def jac(x):
        interior_coords = x.reshape(-1, 2)
        interior_coords = numpy.column_stack([
            interior_coords, numpy.zeros(len(interior_coords))
            ])
        coords = X.copy()
        coords[is_interior_node] = interior_coords

        voropy_mesh = MeshTri(coords, cells, flat_cell_correction=None)
        voropy_mesh, _ = flip_until_delaunay(voropy_mesh)

        grad = numpy.zeros(coords.shape)
        for cell, cc, vol in zip(voropy_mesh.cells['nodes'], voropy_mesh.get_cell_circumcenters(), voropy_mesh.cell_volumes):
            grad[cell] += (coords[cell] - cc) * vol
        grad *= 2 / (gdim+1)

        return grad[is_interior_node, :2].flatten()

    x0 = X[is_interior_node, :2].flatten()

    out = scipy.optimize.minimize(
        f, x0,
        jac=jac,
        method='CG',
        tol=1.0e-5
        )
    assert out.success

    interior_coords = out.x.reshape(-1, 2)
    interior_coords = numpy.column_stack([
        interior_coords, numpy.zeros(len(interior_coords))
        ])
    coords = X.copy()
    coords[is_interior_node] = interior_coords

    mesh = MeshTri(coords, cells, flat_cell_correction=None)
    mesh, _ = flip_until_delaunay(mesh)

    if verbose:
        print('\nBefore:' + 35*' ' + 'After:')
        print_stats([
            initial_stats,
            gather_stats(mesh),
            ])

    return mesh.node_coords, mesh.cells['nodes']

from __future__ import annotations

import re
from typing import Callable

import meshplex
import numpy as np
from numpy.typing import ArrayLike

from . import cpt, cvt, laplace, odt
from .helpers import print_stats

methods = {
    "lloyd": cvt.lloyd,
    "cvt-diaognal": cvt.lloyd,
    "cvt-block-diagonal": cvt.block_diagonal,
    "cvt-full": cvt.full,
    #
    "cpt-linear-solve": cpt.linear_solve,
    "cpt-fixed-point": cpt.fixed_point,
    "cpt-quasi-newton": cpt.quasi_newton,
    #
    "laplace": laplace,
    #
    "odt-fixed-point": odt.fixed_point,
}


def _normalize_method(name: str) -> str:
    # Normalize the method name, e.g.,
    #   ODT  (block diagonal) -> odt-block-diagonal
    return "-".join(
        filter(lambda item: item != "", re.split("-| |\\(|\\)", name.lower()))
    )


def get_new_points(mesh: meshplex.MeshTri, method: str):
    return methods[_normalize_method(method)].get_new_points(mesh)


def optimize(mesh: meshplex.MeshTri, method: str, *args, **kwargs):
    method = _normalize_method(method)

    # Special treatment for ODT. We're using scipy.optimize there.
    if method[:3] == "odt" and method[4:] != "fixed-point":
        min_method = method[4:]
        if "omega" in kwargs:
            assert kwargs["omega"] == 1.0
            kwargs.pop("omega")
        return odt.nonlinear_optimization(mesh, min_method, *args, **kwargs)

    if method not in methods:
        raise KeyError(
            f"Illegal method {method}. Choose one of {', '.join(methods.keys())}."
        )

    return _optimize(methods[method].get_new_points, mesh, *args, **kwargs)


def optimize_points_cells(X: ArrayLike, cells: ArrayLike, method: str, *args, **kwargs):
    cells = np.asarray(cells)
    if cells.shape[1] == 2:
        # line mesh
        mesh = meshplex.Mesh(X, cells)
    else:
        assert cells.shape[1] == 3
        mesh = meshplex.MeshTri(X, cells)
    optimize(mesh, method, *args, **kwargs)
    return mesh.points, mesh.cells("points")


def _optimize(
    get_new_points: Callable,
    mesh: meshplex.MeshTri,
    tol: float,
    max_num_steps: int,
    omega: float = 1.0,
    method_name: str | None = None,
    verbose: bool = False,
    callback: Callable | None = None,
    step_filename_format: str | None = None,
    implicit_surface=None,
    implicit_surface_tol: float = 1.0e-10,
    boundary_step: Callable | None = None,
):
    k = 0

    if verbose:
        print("\nBefore:")
        print_stats(mesh)
    if step_filename_format:
        mesh.save(
            step_filename_format.format(k),
            show_coedges=False,
            show_axes=False,
            cell_quality_coloring=("viridis", 0.0, 1.0, False),
        )

    if callback:
        callback(k, mesh)

    # mesh.write("out0.vtk")
    if hasattr(mesh, "flip_until_delaunay"):
        mesh.flip_until_delaunay()
    # mesh.write("out1.vtk")

    while True:
        k += 1
        new_points = get_new_points(mesh)

        # Move boundary points to the domain boundary, if given. If not just move the
        # points back to their original positions.
        idx = mesh.is_boundary_point
        if boundary_step is None:
            # Reset boundary points to their original positions.
            new_points[idx] = mesh.points[idx]
        else:
            # Move all boundary points back to the boundary.
            new_points[idx] = boundary_step(new_points[idx].T).T

        diff = omega * (new_points - mesh.points)

        # Some methods are stable (CPT), others can break down if the mesh isn't very
        # smooth. A break-down manifests, for example, in a step size that lets cells
        # become completely flat or even "overshoot". After that, anything can happen.
        # To prevent this, restrict the maximum step size to half of the minimum the
        # incircle radius of all adjacent cells. This makes sure that triangles cannot
        # "flip".
        # <https://stackoverflow.com/a/57261082/353337>
        max_step = np.full(mesh.points.shape[0], np.inf)
        np.minimum.at(
            max_step,
            mesh.cells("points").reshape(-1),
            np.repeat(mesh.cell_inradius, mesh.cells("points").shape[1]),
        )
        max_step *= 0.5
        #
        step_lengths = np.sqrt(
            np.einsum(
                "ij,ij->i",
                diff.reshape(diff.shape[0], -1),
                diff.reshape(diff.shape[0], -1),
            )
        )

        # alpha = np.min(max_step / step_lengths)
        # alpha = np.min([alpha, 1.0])
        # diff *= alpha
        idx = step_lengths > max_step
        diff[idx] = (diff[idx].T * (max_step[idx] / step_lengths[idx])).T

        new_points = mesh.points + diff

        # project all points back to the surface, if any
        if implicit_surface is not None:
            fval = implicit_surface.f(new_points.T)
            while np.any(np.abs(fval) > implicit_surface_tol):
                grad = implicit_surface.grad(new_points.T)
                grad_dot_grad = np.einsum("ij,ij->j", grad, grad)
                # The step is chosen in the direction of the gradient with a step size
                # such that, if the function was linear, the boundary (fval=0) would be
                # hit in one step.
                new_points -= (grad * (fval / grad_dot_grad)).T
                # compute new value
                fval = implicit_surface.f(new_points.T)

        mesh.points = new_points
        if hasattr(mesh, "flip_until_delaunay"):
            mesh.flip_until_delaunay()
        # mesh.show(control_volume_centroid_color="C1")
        # mesh.show()

        # Abort the loop if the update was small
        diff_norm_2 = np.einsum(
            "ij,ij->i", diff.reshape(diff.shape[0], -1), diff.reshape(diff.shape[0], -1)
        )
        is_final = np.all(diff_norm_2 < tol**2) or k >= max_num_steps

        if is_final or step_filename_format:
            if is_final:
                info = f"{k} steps"
                if method_name is not None:
                    if abs(omega - 1.0) > 1.0e-10:
                        method_name += f", relaxation parameter {omega}"
                    info += " of " + method_name

                if verbose:
                    print(f"\nFinal ({info}):")
                    print_stats(mesh)
            if step_filename_format:
                mesh.save(
                    step_filename_format.format(k),
                    show_coedges=False,
                    show_axes=False,
                    cell_quality_coloring=("viridis", 0.0, 1.0, False),
                )
        if callback:
            callback(k, mesh)

        if is_final:
            break

    return k, np.max(np.sqrt(diff_norm_2))

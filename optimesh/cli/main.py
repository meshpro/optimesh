# -*- coding: utf-8 -*-
#
import argparse
import math
import sys

import meshio
import numpy

from ..__about__ import __version__
from .. import cpt
from .. import cvt
from .. import odt


def _get_parser():
    parser = argparse.ArgumentParser(description="Mesh smoothing/optimization.")

    parser.add_argument(
        "input_file", metavar="INPUT_FILE", type=str, help="Input mesh file"
    )

    parser.add_argument(
        "output_file", metavar="OUTPUT_FILE", type=str, help="Output mesh file"
    )

    parser.add_argument(
        "--method",
        "-m",
        required=True,
        choices=[
            "cpt-dp",
            "cpt-uniform-fp",
            "cpt-uniform-qn",
            #
            "cvt-uniform-lloyd",
            "cvt-uniform-lloyd2",
            "cvt-uniform-qnb",
            "cvt-uniform-qnf",
            #
            "odt-dp-fp",
            "odt-uniform-fp",
            "odt-uniform-bfgs",
        ],
        help="smoothing method",
    )

    parser.add_argument(
        "--max-num-steps",
        "-n",
        metavar="MAX_NUM_STEPS",
        type=int,
        default=math.inf,
        help="maximum number of steps (default: infinity)",
    )

    parser.add_argument(
        "--tolerance",
        "-t",
        metavar="TOL",
        default=0.0,
        type=float,
        help="convergence criterion (method dependent, default: 0.0)",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="verbose output (default: false)",
    )

    parser.add_argument(
        "--step-filename-format",
        "-f",
        metavar="FMT",
        default=None,
        help=(
            "filename format for mesh at every step "
            "(e.g., `step{:3d}.vtk`, default: None)"
        ),
    )

    parser.add_argument(
        "--subdomain-field-name",
        "-s",
        metavar="SUBDOMAIN",
        default=None,
        help="name of the subdomain field in in the input file (default: None)",
    )

    parser.add_argument(
        "--version",
        "-v",
        help="display version information",
        action="version",
        version="%(prog)s {}, Python {}".format(__version__, sys.version),
    )
    return parser


def main(argv=None):
    parser = _get_parser()
    args = parser.parse_args(argv)

    if not (args.max_num_steps or args.tolerance):
        parser.error("At least one of --max-num_steps or --tolerance required.")

    mesh = meshio.read(args.input_file)

    # TODO remove?
    if mesh.points.shape[1] == 3:
        assert numpy.all(numpy.abs(mesh.points[:, 2]) < 1.0e-13)
        mesh.points = mesh.points[:, :2]

    if args.subdomain_field_name:
        field = mesh.cell_data["triangle"][args.subdomain_field_name]
        subdomain_idx = numpy.unique(field)
        cell_sets = [idx == field for idx in subdomain_idx]
    else:
        cell_sets = [numpy.ones(mesh.cells["triangle"].shape[0], dtype=bool)]

    cells = mesh.cells["triangle"]

    method = {
        "cpt-dp": cpt.linear_solve_density_preserving,
        "cpt-uniform-fp": cpt.fixed_point_uniform,
        "cpt-uniform-qn": cpt.quasi_newton_uniform,
        #
        "cvt-uniform-lloyd": cvt.fixed_point_uniform,
        "cvt-uniform-lloyd2": cvt.quasi_newton_uniform2,
        "cvt-uniform-qnb": cvt.quasi_newton_uniform_blocks,
        "cvt-uniform-qnf": cvt.quasi_newton_uniform_full,
        #
        "odt-dp-fp": odt.fixed_point_density_preserving,
        "odt-uniform-fp": odt.fixed_point_uniform,
        "odt-uniform-bfgs": odt.nonlinear_optimization_uniform,
    }[args.method]

    for cell_idx in cell_sets:
        X, cls = method(
            mesh.points,
            cells[cell_idx],
            args.tolerance,
            args.max_num_steps,
            verbose=args.verbose,
            step_filename_format=args.step_filename_format,
        )
        cells[cell_idx] = cls

    if X.shape[1] != 3:
        X = numpy.column_stack([X[:, 0], X[:, 1], numpy.zeros(X.shape[0])])

    meshio.write_points_cells(
        args.output_file,
        X,
        {"triangle": cells},
        # point_data=mesh.point_data,
        # cell_data=mesh.cell_data,
    )
    return

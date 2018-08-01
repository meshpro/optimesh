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
            "cvt-uniform-fp",
            #
            "odt-dp-fp",
            "odt-uniform-fp",
            "odt-uniform-no",
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
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        help="verbosity level (default: 1)",
        default=1,
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

    for cell_idx in cell_sets:
        if args.method == "cpt-dp":
            X, cls = cpt.linear_solve_density_preserving(
                mesh.points,
                cells[cell_idx],
                args.tolerance,
                args.max_num_steps,
                step_filename_format=args.step_filename_format,
                verbosity=args.verbosity,
            )
        elif args.method == "cpt-uniform-fp":
            X, cls = cpt.fixed_point_uniform(
                mesh.points,
                cells[cell_idx],
                args.tolerance,
                args.max_num_steps,
                step_filename_format=args.step_filename_format,
                verbosity=args.verbosity,
            )
        elif args.method == "cpt-uniform-qn":
            X, cls = cpt.quasi_newton_uniform(
                mesh.points,
                cells[cell_idx],
                args.tolerance,
                args.max_num_steps,
                step_filename_format=args.step_filename_format,
                verbosity=args.verbosity,
            )
        elif args.method == "cvt-uniform-fp":
            X, cls = cvt.fixed_point_uniform(
                mesh.points,
                cells[cell_idx],
                args.tolerance,
                args.max_num_steps,
                verbosity=args.verbosity,
                fcc_type="boundary",
                step_filename_format=args.step_filename_format,
            )
        elif args.method == "odt-dp-fp":
            X, cls = odt.fixed_point_density_preserving(
                mesh.points,
                cells[cell_idx],
                args.tolerance,
                args.max_num_steps,
                step_filename_format=args.step_filename_format,
                verbosity=args.verbosity,
            )
        elif args.method == "odt-uniform-fp":
            X, cls = odt.fixed_point_uniform(
                mesh.points,
                cells[cell_idx],
                args.tolerance,
                args.max_num_steps,
                step_filename_format=args.step_filename_format,
                verbosity=args.verbosity,
            )
        else:
            assert args.method == "odt-uniform-no", "Illegal method {}".format(
                args.method
            )
            X, cls = odt.nonlinear_optimization_uniform(
                mesh.points,
                cells[cell_idx],
                args.tolerance,
                args.max_num_steps,
                step_filename_format=args.step_filename_format,
                verbosity=args.verbosity,
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

# -*- coding: utf-8 -*-
#
import argparse
import meshio
import numpy

from .__about__ import __version__
from .laplace import laplace
from .lloyd import lloyd, lloyd_submesh
from .odt import odt
from . import chen_holst


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
        choices=["laplace", "lloyd", "odt", "chen_odt", "chen_cpt"],
        help="smoothing method",
    )

    parser.add_argument(
        "--max-num-steps",
        "-n",
        metavar="MAX_NUM_STEPS",
        type=int,
        required=True,
        help="maximum number of steps",
    )

    parser.add_argument(
        "--tolerance",
        "-t",
        metavar="TOL",
        type=float,
        required=True,
        help="convergence criterion (method dependent)",
    )

    parser.add_argument(
        "--verbosity", choices=[0, 1, 2], help="verbosity level (default: 1)", default=1
    )

    # parser.add_argument(
    #     "--output-step-filetype",
    #     "-s",
    #     dest="output_steps_filetype",
    #     default=None,
    #     help="write mesh after each Lloyd step",
    # )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s " + ("(version {})".format(__version__)),
    )
    return parser


def main(argv=None):
    parser = _get_parser()
    args = parser.parse_args(argv)

    mesh = meshio.read(args.input_file)

    if mesh.points.shape[1] == 3:
        assert numpy.all(numpy.abs(mesh.points[:, 2]) < 1.0e-13)
        mesh.points = mesh.points[:, :2]

    if args.method == "laplace":
        X, cells = laplace(
            mesh.points,
            mesh.cells["triangle"],
            args.tolerance,
            args.max_num_steps,
            verbosity=args.verbosity,
        )
    elif args.method == "odt":
        X, cells = odt(
            mesh.points,
            mesh.cells["triangle"],
            verbosity=args.verbosity,
            tol=args.tolerance,
        )
    elif args.method == "lloyd":
        # X, cells = lloyd(
        #     mesh.points,
        #     mesh.cells["triangle"],
        #     args.tolerance,
        #     args.max_num_steps,
        #     verbosity=args.verbosity,
        #     fcc_type="boundary",
        # )

        # TODO actual lloyd
        submesh_bools = {0: numpy.ones(len(mesh.cells["triangle"]), dtype=bool)}
        X, cells = lloyd_submesh(
            mesh.points,
            mesh.cells["triangle"],
            args.tolerance,
            args.max_num_steps,
            submesh_bools,
            verbosity=args.verbosity,
            fcc_type="boundary",
        )
    elif args.method == "chen_odt":
        X, cells = chen_holst.odt(
            mesh.points,
            mesh.cells["triangle"],
            args.tolerance,
            args.max_num_steps,
            verbosity=args.verbosity,
        )
    else:
        assert False

    if X.shape[1] != 3:
        X = numpy.column_stack([X[:, 0], X[:, 1], numpy.zeros(X.shape[0])])

    meshio.write_points_cells(args.output_file, X, {"triangle": cells})
    return

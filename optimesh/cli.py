# -*- coding: utf-8 -*-
#
import argparse
import meshio
import numpy

from .__about__ import __version__
from .odt import odt


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

    print(args)
    mesh = meshio.read(args.input_file)

    if mesh.points.shape[1] == 3:
        assert numpy.all(numpy.abs(mesh.points[:, 2]) < 1.0e-13)
        mesh.points = mesh.points[:, :2]

    if args.method == "odt":
        X, cells = odt(mesh.points, mesh.cells["triangle"], verbosity=1, tol=1.0e-5)
    else:
        assert False

    meshio.write_points_cells(args.output_file, X, {"triangle": cells})
    return

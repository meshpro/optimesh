# -*- coding: utf-8 -*-
#
import argparse
import sys

import meshio
from meshplex import MeshTri

from ..__about__ import __version__
from ..helpers import print_stats


def _get_parser():
    parser = argparse.ArgumentParser(description="Display mesh information.")

    parser.add_argument(
        "input_file", metavar="INPUT_FILE", type=str, help="Input mesh file"
    )

    parser.add_argument(
        "--version",
        "-v",
        help="display version information",
        action="version",
        version="%(prog)s {}, Python {}".format(__version__, sys.version),
    )
    return parser


def info(argv=None):
    parser = _get_parser()
    args = parser.parse_args(argv)

    mesh = meshio.read(args.input_file)

    cells = mesh.cells["triangle"]

    print("Number of points: {}".format(mesh.points.shape[0]))
    print("Number of elements:")
    for key, value in mesh.cells.items():
        print("  {}: {}".format(key, value.shape[0]))

    mesh = MeshTri(mesh.points, cells)
    print_stats(mesh)

    return

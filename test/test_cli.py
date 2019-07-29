# -*- coding: utf-8 -*-
#
import pytest

import optimesh
from helpers import download_mesh


@pytest.mark.parametrize(
    "options",
    [
        ["--method", "laplace"],
        #
        ["--method", "cpt-dp"],
        ["--method", "cpt-uniform-fp"],
        ["--method", "cpt-uniform-qn"],
        #
        ["--method", "lloyd"],
        ["--method", "lloyd", "--omega", "2.0"],
        ["--method", "cvt-uniform-qnb"],
        ["--method", "cvt-uniform-qnf", "--omega", "0.9"],
        #
        ["--method", "odt-dp-fp"],
        ["--method", "odt-uniform-fp"],
        ["--method", "odt-uniform-bfgs"],
    ],
)
def test_cli(options):
    input_file = download_mesh(
        # "circle.vtk", "614fcabc0388e1b43723ac64f8555ef52ee4ddda1466368c450741eb"
        "pacman.vtk",
        "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6",
    )
    output_file = "out.vtk"
    optimesh.cli.main([input_file, output_file, "-t", "1.0e-5", "-n", "5"] + options)
    return


def test_info():
    input_file = download_mesh(
        "pacman.vtk", "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6"
    )
    optimesh.cli.info([input_file])
    return


if __name__ == "__main__":
    test_cli("odt")

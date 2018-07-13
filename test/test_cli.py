# -*- coding: utf-8 -*-
#
import pytest

import optimesh

from helpers import download_mesh


@pytest.mark.parametrize(
    "method",
    ["laplace", "lloyd", "odt-fp", "odt-no", "cpt-fp", "cpt-qn"],
)
def test_cli(method):
    input_file = download_mesh(
        # "circle.vtk", "614fcabc0388e1b43723ac64f8555ef52ee4ddda1466368c450741eb"
        "pacman.vtk",
        "19a0c0466a4714b057b88e339ab5bd57020a04cdf1d564c86dc4add6",
    )
    output_file = "out.vtk"
    optimesh.cli.main(
        [input_file, output_file, "--method", method, "-t", "1.0e-5", "-n", "100"]
    )
    return


if __name__ == "__main__":
    test_cli("odt")

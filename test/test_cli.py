# -*- coding: utf-8 -*-
#
import pytest

import optimesh

from helpers import download_mesh


@pytest.mark.parametrize("method", ["laplace", "lloyd", "odt", "chen_odt", "chen_cpt"])
def test_pacman(method):
    input_file = download_mesh(
        "pacman.msh", "601a51e53d573ff58bfec96aef790f0bb6c531a221fd7841693eaa20"
    )
    output_file = "out.vtk"
    optimesh.cli.main(
        [input_file, output_file, "--method", method, "-t", "1.0e-5", "-n", "100"]
    )
    return


if __name__ == "__main__":
    test_pacman("odt")

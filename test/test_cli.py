# -*- coding: utf-8 -*-
#
import numpy

import meshio
import optimesh

from helpers import download_mesh


def test_pacman(num_steps=10):
    input_file = download_mesh(
        "pacman.msh", "601a51e53d573ff58bfec96aef790f0bb6c531a221fd7841693eaa20"
    )

    output_file = "out.vtk"

    optimesh.cli.main([input_file, output_file, "--method", "odt", "-t", "1.0e-5"])

    mesh = meshio.read(output_file)

    # Test if we're dealing with the mesh we expect.
    nc = mesh.node_coords.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    ref = 1917.9950540725958
    assert abs(norm1 - ref) < tol * ref
    ref = 74.99386491032608
    assert abs(norm2 - ref) < tol * ref
    ref = 5.0
    assert abs(normi - ref) < tol * ref
    return


if __name__ == "__main__":
    test_pacman()

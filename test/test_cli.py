import os.path

import pytest

import optimesh


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
    this_dir = os.path.dirname(os.path.realpath(__file__))
    input_file = os.path.join(this_dir, "meshes", "pacman.vtk")
    output_file = "out.vtk"
    optimesh.cli.main([input_file, output_file, "-t", "1.0e-5", "-n", "5"] + options)
    return


def test_info():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    input_file = os.path.join(this_dir, "meshes", "pacman.vtk")
    optimesh.cli.info([input_file])
    return


if __name__ == "__main__":
    test_cli("odt")

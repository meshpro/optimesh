import pathlib

import pytest

import optimesh

this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "options",
    [
        ["--method", "laplace"],
        #
        ["--method", "cpt-linear-solve"],
        ["--method", "cpt-fixed-point"],
        ["--method", "cpt-quasi-newton"],
        #
        ["--method", "lloyd"],
        ["--method", "lloyd", "--omega", "2.0"],
        ["--method", "cvt-block-diagonal"],
        ["--method", "cvt-full", "--omega", "0.9"],
        #
        ["--method", "odt-fixed-point"],
        ["--method", "odt-bfgs"],
    ],
)
def test_cli(options):
    input_file = this_dir / "meshes" / "pacman.vtk"
    output_file = "out.vtk"
    optimesh.cli.main(
        [str(input_file), output_file, "-t", "1.0e-5", "-n", "5"] + options
    )


def test_info():
    input_file = this_dir / "meshes" / "pacman.vtk"
    optimesh.cli.info([str(input_file)])


if __name__ == "__main__":
    test_cli("odt")

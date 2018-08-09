# -*- coding: utf-8 -*-
#
import pytest

import optimesh

from meshes import simple1, pacman

import helpers


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 4.985355657854027, 2.1179164560036154, 1.0),
        (pacman, 1.9391197406035919e+03, 7.5949649769908106e+01, 5.0),
    ],
)
def test_cvt_lloyd(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.cvt.quasi_newton_uniform_lloyd(X, cells, 1.0e-2, 100, verbose=False)

    # Assert that we're dealing with the mesh we expect.
    helpers.assert_norms(X, [ref1, ref2, refi], 1.0e-12)
    return


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 4.9983732074913103, 2.1209374941155565, 1.0),
        (pacman, 1.9366254113952259e+03, 7.5925183214072376e+01, 5.0),
    ],
)
def test_cvt_lloyd2(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.cvt.quasi_newton_uniform_lloyd(X, cells, 1.0e-2, 100, omega=2.0)

    # Assert that we're dealing with the mesh we expect.
    helpers.assert_norms(X, [ref1, ref2, refi], 1.0e-12)
    return


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 4.9968942224093542e+00, 2.1205904527427726e+00, 1.0),
        (pacman, 1.9366979037816668e+03, 7.5929224540258218e+01, 5.0),
    ],
)
def test_cvt_qnb(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.cvt.quasi_newton_uniform_blocks(X, cells, 1.0e-2, 100)

    # Assert that we're dealing with the mesh we expect.
    helpers.assert_norms(X, [ref1, ref2, refi], 1.0e-12)
    return


@pytest.mark.parametrize(
    "mesh, ref1, ref2, refi",
    [
        (simple1, 4.9968942224093542e+00, 2.1205904527427726e+00, 1.0),
        (pacman, 1.9334171657802549e+03, 7.5827921849418885e+01, 5.0),
    ],
)
def test_cvt_qnf(mesh, ref1, ref2, refi):
    X, cells = mesh()

    X, cells = optimesh.cvt.quasi_newton_uniform_full(X, cells, 1.0e-2, 100, omega=0.9)

    # Assert that we're dealing with the mesh we expect.
    helpers.assert_norms(X, [ref1, ref2, refi], 1.0e-12)
    return


if __name__ == "__main__":
    test_cvt_lloyd(pacman, 1939.1198108068188, 75.94965207932323, 5.0)
    # test_cvt_lloyd(simple1, 4.985355657854027, 2.1179164560036154, 1.0)

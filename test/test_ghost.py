# -*- coding: utf-8 -*-
#
import numpy

import optimesh


def test_ghost_flip():
    X = numpy.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = numpy.array([[0, 1, 3], [1, 2, 3]])

    mesh = optimesh.cvt.GhostedMesh(X, cells)

    ref = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    assert numpy.all(numpy.abs(mesh.node_coords - ref) < 1.0e-10)

    ref = numpy.array(
        [[0, 1, 3], [1, 2, 3], [4, 2, 3], [5, 3, 0], [6, 0, 1], [7, 1, 2]]
    )
    assert numpy.all(mesh.cells["nodes"] == ref)
    return


if __name__ == "__main__":
    test_ghost_flip()

# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy

import optimesh

from meshes import circle_random


def test_comparison():
    X, cells = circle_random()
    X = X[:, :2]

    # Do one Lloyd step to avoid too crazy meshes.
    X, cells = optimesh.cpt.fixed_point_uniform(X, cells, 0.0, 2, verbosity=0)

    num_steps = 100
    d = {
        "cpt-uniform-fp": optimesh.cpt.fixed_point_uniform,
        "cpt-uniform-qn": optimesh.cpt.quasi_newton_uniform,
        #
        "cvt-uniform-fp": optimesh.cvt.fixed_point_uniform,
        "cvt-uniform-qn": optimesh.cvt.quasi_newton_uniform,
        #
        "odt-uniform-fp": optimesh.odt.fixed_point_uniform,
        "odt-uniform-bfgs": optimesh.odt.nonlinear_optimization_uniform,
    }

    avg_quality = numpy.empty((len(d), num_steps + 1))

    for i, method in enumerate(d.values()):

        def callback(k, mesh):
            avg_quality[i, k] = numpy.average(mesh.triangle_quality)
            return

        X_in = X.copy()
        cells_in = cells.copy()

        method(X_in, cells_in, 0.0, num_steps, callback=callback, verbosity=0)

    # sort by best final quality
    idx = numpy.argsort(avg_quality[:, -1])[::-1]

    labels = list(d.keys())
    sorted_labels = [labels[i] for i in idx]

    for label, values in zip(sorted_labels, avg_quality[idx]):
        plt.plot(values, "-", label=label)

    plt.xlim(0, num_steps)
    plt.ylim(0.9, 1.0)
    plt.grid()
    plt.xlabel("step")
    plt.ylabel("average cell quality")
    plt.legend()

    plt.savefig("comparison.svg", transparent=True, bbox_inches="tight")
    # plt.show()
    return


if __name__ == "__main__":
    test_comparison()

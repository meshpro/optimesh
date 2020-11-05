import matplotlib.pyplot as plt
import numpy
from meshes import circle_random

import optimesh


def test_comparison():
    X, cells = circle_random(40, 1.0)

    # Do a few steps of a robust method to avoid too crazy meshes.
    tol = 0.0
    n = 10
    # X, cells = optimesh.cpt.fixed_point_uniform(X, cells, tol, n)
    # X, cells = optimesh.odt.fixed_point_uniform(X, cells, tol, n)
    X, cells = optimesh.cvt.quasi_newton_uniform_lloyd(X, cells, tol, n, omega=2.0)

    # from meshplex import MeshTri
    # mesh = MeshTri(X, cells)
    # mesh.write("out.vtk")
    # exit(1)

    num_steps = 50
    d = {
        "cpt-uniform-fp": optimesh.cpt.fixed_point_uniform,
        "cpt-uniform-qn": optimesh.cpt.quasi_newton_uniform,
        #
        "cvt-uniform-lloyd": optimesh.cvt.quasi_newton_uniform_lloyd,
        "cvt-uniform-lloyd(2.0)": optimesh.cvt.quasi_newton_uniform_lloyd,
        "cvt-uniform-qnb": optimesh.cvt.quasi_newton_uniform_blocks,
        "cvt-uniform-qnf": optimesh.cvt.quasi_newton_uniform_full,
        #
        "odt-uniform-fp": optimesh.odt.fixed_point_uniform,
        "odt-uniform-bfgs": optimesh.odt.nonlinear_optimization_uniform,
    }

    avg_quality = numpy.empty((len(d), num_steps + 1))

    for i, (name, method) in enumerate(d.items()):

        def callback(k, mesh):
            avg_quality[i, k] = numpy.average(mesh.q_radius_ratio)
            return

        X_in = X.copy()
        cells_in = cells.copy()

        if name == "cvt-uniform-lloyd(2.0)":
            method(X_in, cells_in, 0.0, num_steps, omega=2.0, callback=callback)
        else:
            method(X_in, cells_in, 0.0, num_steps, callback=callback)

    # sort by best final quality
    idx = numpy.argsort(avg_quality[:, -1])[::-1]

    labels = list(d.keys())
    sorted_labels = [labels[i] for i in idx]

    for label, values in zip(sorted_labels, avg_quality[idx]):
        plt.plot(values, "-", label=label)

    plt.xlim(0, num_steps)
    plt.ylim(0.93, 1.0)
    plt.grid()
    plt.xlabel("step")
    plt.ylabel("average cell quality")
    plt.legend(bbox_to_anchor=(1.03, 1), loc="upper left", borderaxespad=0)

    plt.savefig("comparison.svg", transparent=True, bbox_inches="tight")
    # plt.show()
    return


if __name__ == "__main__":
    test_comparison()

import dufte
import matplotlib.pyplot as plt
import numpy as np

import optimesh

from .meshes import circle_random


def test_comparison():
    plt.style.use(dufte.style)

    X, cells = circle_random(40, 1.0)

    # Do a few steps of a robust method to avoid too crazy meshes.
    tol = 0.0
    n = 10
    # X, cells = optimesh.cpt.fixed_point_uniform(X, cells, tol, n)
    # X, cells = optimesh.odt.fixed_point_uniform(X, cells, tol, n)
    X, cells = optimesh.optimize_points_cells(X, cells, "lloyd", tol, n, omega=2.0)

    # from meshplex import MeshTri
    # mesh = MeshTri(X, cells)
    # mesh.write("out.vtk")
    # exit(1)

    num_steps = 50
    names = [
        "cpt-fixed-point",
        "cpt-quasi-newton",
        #
        "lloyd",
        "lloyd(2.0)",
        "cvt-block-diagonal",
        "cvt-full",
        #
        "odt-fixed-point",
        "odt-bfgs",
    ]

    avg_quality = np.empty((len(names), num_steps + 1))

    for i, name in enumerate(names):

        def callback(k, mesh):
            avg_quality[i, k] = np.average(mesh.q_radius_ratio)
            return

        X_in = X.copy()
        cells_in = cells.copy()

        if name == "lloyd(2.0)":
            optimesh.optimize_points_cells(
                X_in, cells_in, "lloyd", 0.0, num_steps, omega=2.0, callback=callback
            )
        else:
            optimesh.optimize_points_cells(
                X_in, cells_in, name, 0.0, num_steps, callback=callback
            )

    # sort by best final quality
    idx = np.argsort(avg_quality[:, -1])[::-1]

    sorted_labels = [names[i] for i in idx]

    for i, label, values in zip(idx, sorted_labels, avg_quality[idx]):
        plt.plot(values, "-", label=label, zorder=i)

    plt.xlim(0, num_steps)
    plt.ylim(0.93, 1.0)
    plt.xlabel("step")
    plt.title("average cell quality")
    dufte.legend()

    plt.savefig("comparison.svg", transparent=True, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    test_comparison()

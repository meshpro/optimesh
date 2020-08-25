import meshzoo

import optimesh


def test_surface():
    points, cells = meshzoo.tetra_sphere(20)
    # points, cells = meshzoo.octa_sphere(10)
    # points, cells = meshzoo.icosa_sphere(10)

    class Sphere:
        def f(self, x):
            return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

        def grad(self, x):
            return -2 * x

    # points, cells = optimesh.cpt.fixed_point_uniform(
    # points, cells = optimesh.odt.fixed_point_uniform(
    points, cells = optimesh.cvt.quasi_newton_uniform_full(
        points,
        cells,
        1.0e-2,
        100,
        verbose=False,
        implicit_surface=Sphere(),
        step_filename_format="out{:03d}.vtk",
    )


if __name__ == "__main__":
    test_surface()

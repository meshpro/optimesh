# -*- coding: utf-8 -*-
#
"""
Generate animation for fixed-point Laplace. Not included as a command-line option as the
lienar solve is much more efficient.
"""
import meshio
import optimesh

mesh = meshio.read("circle.vtk")
optimesh.laplace.fixed_point(
    mesh.points, mesh.cells["triangle"], 0.0, 50, step_filename_format="step{:03d}.png"
)

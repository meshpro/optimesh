"""
Centroidal Patch Tesselation. Mimics the definition of Centroidal
Voronoi Tessellations for which the generator and centroid of each Voronoi
region coincide. From

Long Chen, Michael Holst,
Efficient mesh optimization schemes based on Optimal Delaunay
Triangulations,
Comput. Methods Appl. Mech. Engrg. 200 (2011) 967–984,
<https://doi.org/10.1016/j.cma.2010.11.007>.
"""

from . import fixed_point, linear_solve, quasi_newton

__all__ = ["linear_solve", "quasi_newton", "fixed_point"]

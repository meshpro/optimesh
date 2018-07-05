import numpy
from scipy.spatial import Delaunay
import meshio


n = 40
boundary_pts = numpy.array(
    [
        [numpy.cos(2 * numpy.pi * k / n), numpy.sin(2 * numpy.pi * k / n)]
        for k in range(n)
    ]
)

# generate random points in circle; <http://mathworld.wolfram.com/DiskPointPicking.html>
numpy.random.seed(123)
m = 200
r = numpy.random.rand(m)
alpha = 2 * numpy.pi * numpy.random.rand(m)

interior_pts = numpy.column_stack([
    numpy.sqrt(r) * numpy.cos(alpha),
    numpy.sqrt(r) * numpy.sin(alpha)
])

pts = numpy.concatenate([boundary_pts, interior_pts])

tri = Delaunay(pts)
pts = numpy.column_stack([pts[:, 0], pts[:, 1], numpy.zeros(pts.shape[0])])
meshio.write_points_cells("circle.vtk", pts, {"triangle": tri.simplices})

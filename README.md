<p align="center">
  <a href="https://github.com/nschloe/optimesh"><img alt="optimesh" src="https://nschloe.github.io/optimesh/optimesh-logo.svg" width="60%"></a>
  <p align="center">Triangular mesh optimization.</p>
</p>

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/optimesh/master.svg?style=flat-square)](https://circleci.com/gh/nschloe/optimesh)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/optimesh.svg?style=flat-square)](https://codecov.io/gh/nschloe/optimesh)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![smooth](https://img.shields.io/badge/smooth-operator-8209ba.svg?style=flat-square)](https://youtu.be/4TYv2PhG89A)
[![PyPi Version](https://img.shields.io/pypi/v/optimesh.svg?style=flat-square)](https://pypi.org/project/optimesh)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/optimesh.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/optimesh)
[![PyPi downloads](https://img.shields.io/pypi/dm/optimesh.svg?style=flat-square)](https://pypistats.org/packages/optimesh)

Several mesh smoothing/optimization methods with one simple interface. optimesh

 * is fast,
 * preserves submeshes,
 * only works for triangular meshes, flat and on a surface, (for now; upvote [this
   issue](https://github.com/nschloe/optimesh/issues/17) if you're interested in
   tetrahedral mesh smoothing), and
 * supports all mesh formats that [meshio](https://github.com/nschloe/meshio) can
   handle.

Install with
```
pip3 install optimesh --user
```
Example call:
```
optimesh in.e out.vtk
```
Output:
![terminal-screenshot](https://nschloe.github.io/optimesh/term-screenshot.png)

The left hand-side graph shows the distribution of angles (the grid line is at the
optimal 60 degrees). The right hand-side graph shows the distribution of simplex
quality, where quality is twice the ratio of circumcircle and incircle radius.

All command-line options are documented at
```
optimesh -h
```

![disk-step0](https://nschloe.github.io/optimesh/disk-step0.png)

The following examples show the various algorithms at work, all starting from the same
randomly generated disk mesh above. The cell coloring indicates quality; dark green is
bad, yellow is good.


#### CVT (centroidal Voronoi tesselation)

![cvt-uniform-lloyd2](https://nschloe.github.io/optimesh/lloyd2.webp) |
![cvt-uniform-qnb](https://nschloe.github.io/optimesh/cvt-uniform-qnb.webp) |
![cvt-uniform-qnf](https://nschloe.github.io/optimesh/cvt-uniform-qnf.webp) |
:------------------------:|:---------------------:|:----:|
uniform-density relaxed [Lloyd's algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) (`--method lloyd --omega 2.0`) | uniform-density quasi-Newton iteration (block-diagonal Hessian, `--method cvt-uniform-qnb`) | uniform-density quasi-Newton iteration (default method, full Hessian, `--method cvt-uniform-qnf`) |

Centroidal Voronoi tessellation smoothing ([Du et al.](#relevant-publications)) is one
of the oldest and most reliable approaches. optimesh provides classical Lloyd smoothing
as well as several variants that result in better meshes.


#### CPT (centroidal patch tesselation)

![cpt-cp](https://nschloe.github.io/optimesh/cpt-dp.png) |
![cpt-uniform-fp](https://nschloe.github.io/optimesh/cpt-uniform-fp.webp) |
![cpt-uniform-qn](https://nschloe.github.io/optimesh/cpt-uniform-qn.webp) |
:-----------------------------------------------------------------------:|:-----------------------------------------------------------------:|:--------------------------------------------------------:|
density-preserving linear solve ([Laplacian smoothing](https://en.wikipedia.org/wiki/Laplacian_smoothing), `--method cpt-dp`) | uniform-density fixed-point iteration (`--method cpt-uniform-fp`) | uniform-density quasi-Newton (`--method cpt-uniform-qn`) |

A smoothing method suggested by [Chen and Holst](#relevant-publications), mimicking CVT
but much more easily implemented. The density-preserving variant leads to the exact same
equation system as Laplacian smoothing, so CPT smoothing can be thought of as a
generalization.

The uniform-density variants are implemented classically as a fixed-point iteration and
as a quasi-Newton method. The latter typically converges faster.


#### ODT (optimal Delaunay tesselation)

![odt-dp-fp](https://nschloe.github.io/optimesh/odt-dp-fp.webp) |
![odt-uniform-fp](https://nschloe.github.io/optimesh/odt-uniform-fp.webp) |
![odt-uniform-bfgs](https://nschloe.github.io/optimesh/odt-uniform-bfgs.webp) |
:--------------------------------------------------------------:|:-----------------------------------------------------------------:|:------------------------------------------------------------------:|
density-preserving fixed-point iteration (`--method odt-dp-fp`) | uniform-density fixed-point iteration (`--method odt-uniform-fp`) | uniform-density BFGS (`--method odt-uniform-bfgs`) |

Optimal Delaunay Triangulation (ODT) as suggested by [Chen and
Holst](#relevant-publications). Typically superior to CPT, but also more expensive to
compute.

Implemented once classically as a fixed-point iteration, once as a nonlinear
optimization method. The latter typically leads to better results.


### Surface mesh smoothing

optimesh also supports optimization of triangular meshes on surfaces which are defined
implicitly by a level set function (e.g., spheres). You'll need to specify the function
and its gradient, so you'll have to do it in Python:
```python
import meshzoo
import optimesh

points, cells = meshzoo.tetra_sphere(20)

class Sphere:
    def f(self, x):
        return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    def grad(self, x):
        return -2 * x

# You can use all methods in optimesh:
# points, cells = optimesh.cpt.fixed_point_uniform(
# points, cells = optimesh.odt.fixed_point_uniform(
points, cells = optimesh.cvt.quasi_newton_uniform_full(
    points, cells, 1.0e-2, 100, verbose=False,
    implicit_surface=Sphere(),
    # step_filename_format="out{:03d}.vtk"
)
```
This code first generates a mediocre mesh on a sphere using
[meshzoo](https://github.com/nschloe/meshzoo/),

![tetra-sphere](https://nschloe.github.io/optimesh/tetra-sphere.png)

and then optimizes. Some results:

![odt-dp-fp](https://nschloe.github.io/optimesh/sphere-cpt.webp) |
![odt-uniform-fp](https://nschloe.github.io/optimesh/sphere-odt.webp) |
![odt-uniform-bfgs](https://nschloe.github.io/optimesh/sphere-cvt.webp) |
:------------:|:----------:|:---------------------------------:|
CPT | ODT | CVT (full Hessian) |


### Which method is best?

From practical experiments, it seems that the CVT smoothing variants, e.g.,
```
optimesh in.vtk out.vtk -m cvt-uniform-qnf
```
give very satisfactory results. (This is also the default method, so you don't need to
specify it explicitly.)  Here is a comparison of all uniform-density methods applied to
the random circle mesh seen above:

<img src="https://nschloe.github.io/optimesh/comparison.svg" width="90%">

(Mesh quality is twice the ratio of incircle and circumcircle radius, with the maximum
being 1.)


### Access from Python

All optimesh functions can also be accessed from Python directly, for example:
```python
import optimesh

X, cells = optimesh.odt.fixed_point_uniform(X, cells, 1.0e-2, 100, verbose=False)
```

### Installation

optimesh is [available from the Python Package
Index](https://pypi.org/project/optimesh/), so simply do
```
pip3 install --upgrade --user optimesh
```
to install or upgrade.

### Relevant publications

 * [Qiang Du, Vance Faber, and Max Gunzburger, _Centroidal Voronoi Tessellations: Applications and Algorithms_,
   SIAM Rev., 41(4), 1999, 637–676.](https://doi.org/10.1137/S0036144599352836)

 * [Yang Liu et al., _On centroidal Voronoi tessellation—Energy smoothness and fast computation_,
   ACM Transactions on Graphics, Volume 28, Issue 4, August 2009.](https://dl.acm.org/citation.cfm?id=1559758)

 * [Long Chen, Michael Holst, _Efficient mesh optimization schemes based on Optimal Delaunay Triangulations_,
   Comput. Methods Appl. Mech. Engrg. 200 (2011) 967–984.](https://doi.org/10.1016/j.cma.2010.11.007)


### Testing

To run the optimesh unit tests, check out this repository and type
```
pytest
```

### License

optimesh is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).

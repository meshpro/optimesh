# optimesh

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/optimesh/master.svg)](https://circleci.com/gh/nschloe/optimesh)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/optimesh.svg)](https://codecov.io/gh/nschloe/optimesh)
[![Codacy grade](https://img.shields.io/codacy/grade/97175bbf62854fcfbfc1f5812ce840f7.svg)](https://app.codacy.com/app/nschloe/optimesh/dashboard)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![smooth](https://img.shields.io/badge/smooth-operator-8209ba.svg)](https://youtu.be/4TYv2PhG89A)
[![PyPi Version](https://img.shields.io/pypi/v/optimesh.svg)](https://pypi.org/project/optimesh)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/optimesh.svg?logo=github&label=Stars)](https://github.com/nschloe/optimesh)

Several mesh smoothing/optimization methods with one simple interface. optimesh

 * is fast,
 * preserves submeshes,
 * only works for triangular meshes (for now), and
 * supports all mesh formats that [meshio](https://github.com/nschloe/meshio) can
   handle.

Install with
```
pip install optimesh
```
Example call:
```
optimesh in.e out.vtk --method lloyd -n 50
```
Output:
![terminal-screenshot](https://nschloe.github.io/optimesh/term-screenshot.png)

The left hand-side graph shows the distribution of angles (the grid line is at the
optimal 60 degrees). The right hand-side graph shows the distribution of simplex
quality, where quality is twice the ratio of circumcircle and incircle radius.

All command-line options are viewed with
```
optimesh -h
```

#### Laplacian smoothing

![laplace-fp](https://nschloe.github.io/optimesh/laplace-fp.png) |
![laplace-ls](https://nschloe.github.io/optimesh/laplace.png) |
:----------------:|:---------------------------------:|
classical Laplace | linear solve (`--method laplace`) |

Classical [Laplacian mesh smoothing](https://en.wikipedia.org/wiki/Laplacian_smoothing)
means moving all (interior) points into the average of their neighbors until an
equilibrium has been reached. The method preserves the mesh density (i.e., small
simplices are not blown up as part of the smoothing).

Instead of a fixed-point iteration, one can do a few linear solves, interleaved with
facet-flipping. This approach converges _much_ faster.


#### CVT (centroidal Voronoi tesselation)

![lloyd](https://nschloe.github.io/optimesh/lloyd.png) |
:---------------:|
`--method lloyd` |

Centroidal Voronoi tessellation smoothing, realized by [Lloyd's
algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm), i.e., points are
iteratively moved into the centroid of their Voronoi cell.  If the topological neighbors
of any node are also the geometrically closest nodes, this is exactly Lloyd's algorithm.
That is fulfilled in many practical cases, but the algorithm can break down if it is
not.


#### CPT (centroidal patch tessalation)

![cpt-fp](https://nschloe.github.io/optimesh/cpt-fp.png) |
![cpt-qn](https://nschloe.github.io/optimesh/cpt-qn.png) |
:----------------------------------------:|:--------------------------------:|
fixed-point iteration (`--method cpt-fp`) | quasi-Newton (`--method cpt-qn`) |

A smooting method suggested by [Chen and Holst](#relevant-publications), mimicking CVT
but much more easily implemented. The density-preserving variant leads to the exact same
equation system as Laplace smoothing, so optimesh only contains the the uniform-density
variant.

Implemented once classically as a fixed-point iteration, once as a quasi-Newton method.
The latter typically leads to better results.


#### ODT (optimal Delaunay tesselation)

![odt-fp](https://nschloe.github.io/optimesh/odt-fp.png) |
![odt-no](https://nschloe.github.io/optimesh/odt-no.png) |
:----------------------------------------:|:------------------------------------------:|
fixed-point iteration (`--method odt-fp`) | nonlinear optimization (`--method odt-no`) |

Optimal Delaunay Triangulation (ODT) as suggested by [Chen and
Holst](#relevant-publications). Typically superior to CPT, but also more expensive to
compute.

Implemented once classically as a fixed-point iteration, once as a nonlinear
optimization method. The latter typically leads to better results.


### Access from Python

All optimesh functions can also be accessed from Python directly, for example:
```python
import optimesh

X, cells = optimesh.odt(X, cells, 1.0e-2, 100, verbosity=1)
```

### Installation

optimesh is [available from the Python Package
Index](https://pypi.org/project/optimesh/), so simply do
```
pip install -U optimesh
```
to install or upgrade. Use `sudo -H` to install as root or the `--user` option
of `pip` to install in `$HOME`.

### Relevant publications

 * [Long Chen, Michael Holst, _Efficient mesh optimization schemes based on Optimal Delaunay Triangulations_,
   Comput. Methods Appl. Mech. Engrg. 200 (2011) 967–984.](https://doi.org/10.1016/j.cma.2010.11.007)


### Testing

To run the optimesh unit tests, check out this repository and type
```
pytest
```

### Distribution
To create a new release

1. bump the `__version__` number,

2. publish to PyPi and tag on GitHub:
    ```
    $ make publish
    ```

### License

optimesh is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).

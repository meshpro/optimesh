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
![laplace-ls](https://nschloe.github.io/optimesh/laplace-ls.png) |
:-------------------------------------:|:-----------------------------------------------------:|
classical Laplace (density-preserving) | linear solve (density-preserving, `--method laplace`) |

Classical [Laplacian mesh smoothing](https://en.wikipedia.org/wiki/Laplacian_smoothing)
means moving all points into the average of their neighbors until an equilibrium has
been reached. The method preserves the mesh density (i.e., small simplices
are not blown up as part of the smoothing).

Instead a fixed-point iteration, one can do a simple linear solve, yielding much faster
convergence.


#### CVT/pseudo-Lloyd smoothing

![lloyd](https://nschloe.github.io/optimesh/lloyd.png) |
:---------------------------------------:|
fixed-point iteration (`--method lloyd`) |

Centroidal Voronoi tessellation smoothing, realized by [Lloyd's
algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm).
If the topological neighbors of any node are also the geometrically closest
nodes, this is exactly Lloyd's algorithm. That is fulfilled in many practical cases, but
the algorithm can break down if it is not.

Assumes a uniform mesh density (for now), so it does _not_ preserve the original mesh
density.


#### CPT smoothing

![cpt-ls](https://nschloe.github.io/optimesh/cpt-ls.png) |
![cpt-fp](https://nschloe.github.io/optimesh/cpt-fp.png) |
![cpt-qn](https://nschloe.github.io/optimesh/cpt-qn.png) |
----------------------------------------------------:|:-----------------------------------------:|:--------------------------------:|
linear solve (density-preserving, `--method cpt-ls`) | fixed-point iteration (`--method cpt-fp`) | quasi-Newton (`--method cpt-qn`) |

Optimal Delaunay Triangulation (ODT) treated as a minimization problem.
Assumes a uniform mesh (for now), so it does _not_ preserve the original mesh density.


#### ODT smoothing

![odt-fp](https://nschloe.github.io/optimesh/odt-fp.png) |
![odt-no](https://nschloe.github.io/optimesh/odt-no.png) |
:----------------------------------------:|:------------------------------------------:|
fixed-point iteration (`--method odt-fp`) | nonlinear optimization (`--method odt-no`) |

Optimal Delaunay Triangulation (ODT) treated as a minimization problem.
Assumes a uniform mesh (for now), so it does _not_ preserve the original mesh density.


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

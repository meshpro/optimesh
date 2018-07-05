# optimesh

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/optimesh/master.svg)](https://circleci.com/gh/nschloe/optimesh)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/optimesh.svg)](https://codecov.io/gh/nschloe/optimesh)
[![Codacy grade](https://img.shields.io/codacy/grade/97175bbf62854fcfbfc1f5812ce840f7.svg)](https://app.codacy.com/app/nschloe/optimesh/dashboard)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![smooth](https://img.shields.io/badge/smooth-yes-8209ba.svg)](https://github.com/nschloe/smoothfit)
[![PyPi Version](https://img.shields.io/pypi/v/optimesh.svg)](https://pypi.org/project/optimesh)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/optimesh.svg?logo=github&label=Stars)](https://github.com/nschloe/optimesh)

```
optimesh -h
```

#### Laplacian smoothing
![laplace](https://nschloe.github.io/optimesh/laplace.png)

Ordinary [Laplacian mesh smoothing](https://en.wikipedia.org/wiki/Laplacian_smoothing).
Fast, preserves the mesh density.
```
optimesh circle.vtk out.vtk --method laplace
```

#### ODT
Optimal Delaunay Triangulation (ODT) treated as a minimzation problem
```
optimesh circle.vtk out.vtk --method odt
```

#### Chen-Holst
Mesh optimization after [Chen and Holst](#relevant-publications).
ODT
```
optimesh circle.vtk out.vtk --method chen-odt --uniform-density"
```
CPT
```
optimesh circle.vtk out.vtk --method chen-cpt --uniform-density"
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

# Installation

ngsufl is pure Python.  It needs NGSolve and UFL, both of which come as wheels from
PyPI for Linux, macOS and Windows, and Python 3.10 or newer.

## Install

A virtual environment keeps things separate from other installations:

```bash
python3 -m venv ngsufl-env
source ngsufl-env/bin/activate
```

Then

```bash
pip install --upgrade ngsolve fenics-ufl matplotlib
pip install git+https://github.com/JSchoeberl/ngsufl
```

`matplotlib` is only needed for plots.  For the tutorials as notebooks add
`pip install jupyterlab`, and for the interactive NGSolve viewer
`pip install webgui_jupyter_widgets`.

```{note}
ngsufl is not on PyPI yet.  While the repository is private, the second line asks for
a GitHub login with access to it.  With an SSH key use
`pip install git+ssh://git@github.com/JSchoeberl/ngsufl`.
```

## Check

```python
from ngsufl import *

mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "CG", 2)
u, v = TrialFunction(V), TestFunction(V)
x, y = SpatialCoordinate(mesh)

uh = Function(V)
solve(inner(grad(u), grad(v)) * dx == 2 * pi**2 * sin(pi * x) * sin(pi * y) * v * dx,
      uh, bcs=DirichletBC(V, 0, "on_boundary"))
print(errornorm(sin(pi * x) * sin(pi * y), uh))
```

prints `0.000548...`.

## Versions

Tested with NGSolve 6.2.2608, UFL 2026.2 and Python 3.15.  Older NGSolve versions
are untested; `pip install --upgrade ngsolve` gives a current one.

If NGSolve is already installed in another way (a self-built version, the Netgen app),
install only `fenics-ufl` and ngsufl into that Python.  Make sure that one NGSolve is
found: a `PYTHONPATH` that points to another NGSolve takes precedence over the one in
the virtual environment.

## Legacy FEniCS scripts

Nothing else is needed.  Replace `from fenics import *` or `from dolfin import *` by

```python
from ngsufl.fenics import *
```

## From source

```bash
git clone https://github.com/JSchoeberl/ngsufl
cd ngsufl
pip install -e ".[test,docs]"
pytest                             # the test suite, a few seconds
python docs/make_notebooks.py      # execute the tutorials
jupyter-book build docs            # build this book
```

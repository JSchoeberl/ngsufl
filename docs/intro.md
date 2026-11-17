# ngsufl

**The Unified Form Language on the NGSolve finite element core.**

[UFL](https://github.com/FEniCS/ufl) is the form language of FEniCS and Firedrake.
[NGSolve](https://ngsolve.org) is a high order finite element library with its own
geometry kernel, mesh generator and solvers.  ngsufl connects the two: the variational
problem is written in UFL, and NGSolve assembles and solves it.

```python
from ngsufl import *

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 3)

u, v = TrialFunction(V), TestFunction(V)
x = SpatialCoordinate(mesh)
f = 2 * pi**2 * sin(pi * x[0]) * sin(pi * x[1])

uh = Function(V)
solve(inner(grad(u), grad(v)) * dx == f * v * dx, uh, DirichletBC(V, 0, "on_boundary"))
```

The tutorials in this book rework the classic problems of the
[FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/) and the
[DOLFINx demos](https://docs.fenicsproject.org/dolfinx/main/python/demos.html).  The
variational forms are the same UFL one writes there.  Every notebook checks its result
against an exact solution or against the same problem written directly in NGSolve.

```{warning}
ngsufl is experimental and a few days old.  The name and the front end API are provisional.
```

```{tableofcontents}
```

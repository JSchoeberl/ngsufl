# ngsufl

<p align="center"><img src="docs/logo.png" width="380" alt="NGSolve and UFL clinking glasses"></p>

**The Unified Form Language on the NGSolve finite element core.**

Write your variational problem in [UFL](https://github.com/FEniCS/ufl), the form
language of FEniCS and Firedrake, and let [NGSolve](https://ngsolve.org) do the finite
element work.

```python
from ngsufl import *

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 3)

u, v = TrialFunction(V), TestFunction(V)
x, y = SpatialCoordinate(mesh)
f = 2 * pi**2 * sin(pi * x) * sin(pi * y)

a = inner(grad(u), grad(v)) * dx
L = f * v * dx

uh = Function(V)
solve(a == L, uh, bcs=DirichletBC(V, 0, "on_boundary"))

print(errornorm(sin(pi * x) * sin(pi * y), uh))
```

The forms are real UFL objects, so `derivative`, `action`, `adjoint`, `lhs`/`rhs`,
`replace`, `split`, `jump`/`avg`, index notation and the rest of the language work as
documented by UFL.  The spaces, the assembly and the solvers are NGSolve.

## Two dialects, one back end

UFL fixes the forms, not the code around them.  ngsufl offers that code in two styles:

```python
from ngsufl import *           # Firedrake style, the default
from ngsufl.fenics import *    # legacy FEniCS (dolfin) style
```

The default follows Firedrake: mesh constructors and their boundary numbers,
`solve(a == L, u, bcs=..., solver_parameters=..., nullspace=...)`, `Function.interpolate`,
`assemble`, `errornorm`, `VTKFile`, and the plotting functions of `firedrake.pyplot` in
`ngsufl.pyplot`.  The legacy dialect adds what dolfin scripts use: `Expression` with C++
strings, boundary functions `(x, on_boundary)`, `Point`, `solve(a == L, u, bc)`,
`plot`, `File`.  Both are thin layers over the same translator and the same NGSolve
objects.

> **Status: experimental.**  The package is brand new.  The name, the front end API and
> the home of the repository are provisional.

## Tutorials

The [tutorial book](https://jschoeberl.github.io/ngsufl/) reworks the classic FEniCS
tutorial and demo problems: Poisson, membrane, heat equation, nonlinear Poisson,
boundary conditions, convergence, linear elasticity, hyperelasticity, Stokes,
Navier-Stokes, mixed Poisson, discontinuous Galerkin, biharmonic, Cahn-Hilliard and
Maxwell.  Every notebook checks its result against an exact solution or against the
same problem written directly in NGSolve.

## Installation

```bash
pip install --upgrade ngsolve fenics-ufl matplotlib
pip install git+https://github.com/JSchoeberl/ngsufl
```

`matplotlib` is only needed for plots; for the interactive NGSolve viewer add
`webgui_jupyter_widgets`.  Details and a first check are on the
[installation page](https://jschoeberl.github.io/ngsufl/install.html).

## What you get

| | |
|---|---|
| **Language** | all of UFL (the `fenics-ufl` package itself, not a reimplementation) |
| **Elements** | Lagrange, DG, Crouzeix-Raviart, Nedelec 1st and 2nd kind, Raviart-Thomas, BDM, Regge, Hellan-Herrmann-Johnson, Real; any order; triangles, quadrilaterals, tetrahedra, hexahedra; mixed spaces |
| **Measures** | `dx`, `ds`, `dS`; subdomains by number as in Firedrake (`ds(1)`, `ds((3, 4))`) or by NGSolve name (`ds("inlet\|outlet")`, `dx("copper")`) |
| **Geometry** | structured meshes, and any Netgen / OpenCascade geometry with curved elements |
| **Nonlinear problems** | Newton with the Jacobian from UFL's `derivative` or from NGSolve's own differentiation |
| **Solvers** | NGSolve direct and Krylov solvers behind PETSc-style option names; null spaces |
| **Time stepping** | Runge-Kutta and multistep methods of [Irksome](https://www.firedrakeproject.org/Irksome/) through the back end `ngsufl.irksome` |
| **Interoperability** | both ways: `u.gf` is the NGSolve `GridFunction` and `to_cf(expr)` the `CoefficientFunction` of any UFL expression; `from_cf(cf)` puts any NGSolve `CoefficientFunction` into a UFL form |

## How it works

UFL is a symbolic language without a back end.  ngsufl is a further back end next to
FFCx (DOLFINx) and TSFC (Firedrake), with one difference: nothing is compiled.

* A UFL expression tree is walked once and turned into an NGSolve
  `CoefficientFunction` tree.  Compound tensor operators (`inner`, `dot`, `sym`, `det`,
  `inv`, ...) map one to one.
* Index notation is collected into lazy Einstein products and emitted as native NGSolve
  operations where a pattern is recognised (`A[i,j]*B[j,k]` becomes a matrix product),
  and as `Einsum` otherwise.
* `grad`, `div` and `curl` of functions become the differential operators of the NGSolve
  space.  Derivatives of general expressions are first expanded by UFL.  NGSolve applies
  the pullbacks itself, so UFL's reference-cell lowering is not used.
* `u('+')` and `u('-')` become `u` and `u.Other()` on NGSolve's skeleton integrals.
* The quadrature order follows UFL's degree estimation, as in FEniCS.
* A FEniCS element name is a row in a table: `("RT", k)` is `HDiv(order=k-1, RT=True)`,
  `("N1curl", k)` is `HCurl(order=k, type1=True)`, and so on.

## What is different from Firedrake and FEniCS

* **Degrees of freedom are not nodal.**  NGSolve uses hierarchical bases.  Function
  values are available through evaluation, interpolation and projection.  The
  coefficient array (`u.dat.data`, `u.vector()`) holds point values only for lowest
  order spaces.
* **The front end covers the common path**, not all of Firedrake or dolfin.  Mesh tags,
  dof maps, extruded meshes and direct PETSc access are not provided.  The DOLFINx
  interface is not imitated.
* **Real arithmetic only** for now.
* **Serial and shared memory.**  MPI and PETSc solvers are planned through
  [ngsPETSc](https://github.com/NGSolve/ngsPETSc).

## Development

```bash
git clone https://github.com/JSchoeberl/ngsufl
cd ngsufl
pip install -e ".[test,docs]"
pytest
python docs/make_notebooks.py      # execute the tutorials
jupyter-book build docs
```

## License and authors

MIT.  Written by Joachim Schöberl and Claude (Anthropic).  UFL is developed by the
FEniCS project, NGSolve by the NGSolve team.

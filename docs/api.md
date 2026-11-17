# API overview

`from ngsufl import *` exports the whole UFL namespace and the front end below.

## Meshes

`Mesh` wraps an NGSolve mesh, available as `mesh.ngs`.

| | |
|---|---|
| `UnitIntervalMesh(n)`, `IntervalMesh(n, a, b)` | boundaries `left`, `right` |
| `UnitSquareMesh(nx, ny, quadrilateral=False)`, `RectangleMesh(nx, ny, Lx, Ly, ...)` | boundaries `bottom`, `right`, `top`, `left` |
| `UnitCubeMesh(nx, ny, nz, hexahedral=False)`, `BoxMesh(nx, ny, nz, Lx, Ly, Lz, ...)` | boundaries `back` (x=0), `front`, `left` (y=0), `right`, `bottom` (z=0), `top` |
| `Mesh(ngsolve_mesh)`, `Mesh(netgen_mesh)`, `Mesh("file.vol")` | any Netgen or OpenCascade geometry |
| `mesh.curve(order)` | curved elements |
| `mesh.boundary_names`, `mesh.materials`, `mesh.boundary_ids("left\|top")` | names and 1-based numbers |

Subdomains in measures are given by name or regular expression, `ds("left|right")`,
`dx("copper")`, or by 1-based number, `ds(2)`, `ds((1, 3))`.

## Spaces

| | |
|---|---|
| `FunctionSpace(mesh, "Lagrange", 2)` | also `FunctionSpace(mesh, ("Lagrange", 2))` |
| `VectorFunctionSpace(mesh, "Lagrange", 2)`, `TensorFunctionSpace(...)` | |
| `V * Q`, `MixedFunctionSpace([V, Q, R])` | mixed spaces |
| `W.sub(0)`, `V.sub(1)` | component views for boundary conditions |
| `V.dim()`, `V.fes` | number of dofs, the NGSolve space |

Families: `Lagrange` (`P`, `CG`, `Q`), `DG` (`DP`, `DQ`), `CR`, `Real`, `N1curl`, `N2curl`,
`RT`, `BDM`, `Regge`, `HHJ`.  Degrees follow FEniCS.  Extra keyword arguments are passed
to the NGSolve space.

## Functions

| | |
|---|---|
| `u = Function(V, name="u")` | `u.gf` is the NGSolve `GridFunction`, `u.cf` its `CoefficientFunction` |
| `u.interpolate(expr)` | UFL expression, number, tuple, `lambda x: np.sin(x[0])`, or an NGSolve CF |
| `u.assign(w)`, `u.copy()` | |
| `u.at(x, y)`, `u.at(points)` | point values |
| `u.subfunctions`, `u.sub(i)` | components of a mixed function, sharing its data |
| `Constant(value)`, `Constant(mesh, value)` | `c.assign(v)`, `c.value` |
| `from_cf(cf, mesh, degree=1)` | an NGSolve CoefficientFunction as a UFL coefficient |
| `material_function(mesh, {"iron": 1000, "air": 1})` | piecewise constant by material name |
| `to_cf(expr)` | the NGSolve CoefficientFunction of a UFL expression |

## Boundary conditions

`DirichletBC(V, g, where)` with `where` one of `"on_boundary"`, a name or regular
expression, a 1-based number or tuple of numbers, or a callable: vectorised
`lambda x: np.isclose(x[0], 0)` or `lambda x, on_boundary: ...`.  A callable that
marks single points pins vertex values.

## Solving

| | |
|---|---|
| `solve(a == L, u, bcs)` | linear problem |
| `solve(F == 0, u, bcs, J=None)` | Newton; without `J` NGSolve differentiates `F` |
| `LinearVariationalProblem(a, L, u, bcs, constant_jacobian=False)`, `LinearVariationalSolver(problem)` | `constant_jacobian=True` factorises once |
| `NonlinearVariationalProblem(F, u, bcs, J=None)`, `NonlinearVariationalSolver(problem)` | `.solve()`, `.iterations` |
| `assemble(form)` | number, NGSolve vector or NGSolve matrix |
| `errornorm(u_exact, uh, "L2")`, `norm(expr, "H1")` | `L2`, `H1`, `H10`, `Hdiv`, `Hcurl` |
| `project(expr, V)`, `interpolate(expr, V)` | |

Solver options use PETSc names: `ksp_type` (`preonly`, `cg`, `gmres`, `minres`), `pc_type`
(`lu`, `cholesky`, `jacobi`, `none`), `ksp_rtol`, `ksp_max_it`, `ksp_monitor`, `snes_atol`,
`snes_rtol`, `snes_max_it`, `snes_monitor`, `snes_damping`.

The quadrature order follows UFL's degree estimation.  Set it with `dx(degree=4)`.

## Output

| | |
|---|---|
| `plot(u)`, `plot(expr)`, `plot(mesh)` | matplotlib: 1D curve, 2D filled plot, 3D boundary surface with optional `deformation=` |
| `VTKFile("name").write(u, p)` | VTK files |
| `Draw(u)` | NGSolve webgui |

# API overview

`from ngsufl import *` exports the whole UFL namespace and the front end below.  It
follows Firedrake; what goes beyond Firedrake is marked as an *extra*.  The legacy
FEniCS dialect is described at the end.

## Meshes

`Mesh` wraps an NGSolve mesh, available as `mesh.ngs`.

| | boundary numbers | names (extra) |
|---|---|---|
| `UnitIntervalMesh(n)`, `IntervalMesh(n, length)`, `IntervalMesh(n, left, right)` | 1 left, 2 right | `left`, `right` |
| `UnitSquareMesh(nx, ny)`, `SquareMesh(nx, ny, L)`, `RectangleMesh(nx, ny, Lx, Ly, originX=0, originY=0)` | 1: x = originX, 2: x = Lx, 3: y = originY, 4: y = Ly | `left`, `right`, `bottom`, `top` |
| `UnitCubeMesh(nx, ny, nz)`, `CubeMesh(nx, ny, nz, L)`, `BoxMesh(nx, ny, nz, Lx, Ly, Lz)` | 1: x = 0, 2: x = Lx, 3: y = 0, 4: y = Ly, 5: z = 0, 6: z = Lz | `back`, `front`, `left`, `right`, `bottom`, `top` |

Keywords: `quadrilateral=True`, `hexahedral=True`, `diagonal="left"` or `"right"`.  The
parallel keywords of Firedrake (`comm`, `reorder`, `distribution_parameters`, ...) are
accepted and ignored.

| | |
|---|---|
| `Mesh(ngsolve_mesh)`, `Mesh(netgen_mesh)`, `Mesh("file.vol")` | extra: any Netgen or OpenCascade geometry; boundaries by name or by 1-based NGSolve index |
| `mesh.curve(order)` | extra: curved elements |
| `mesh.coordinates` | vertex coordinates as a piecewise linear vector `Function` |
| `mesh.num_cells()`, `mesh.num_vertices()`, `mesh.geometric_dimension()` | |
| `mesh.boundary_names`, `mesh.materials`, `mesh.boundary_ids("left\|top")` | extra |

Subdomains in measures are given by number, `ds(2)`, `ds((1, 3))`, or by name or
regular expression, `ds("left|right")`, `dx("copper")`.

## Spaces

| | |
|---|---|
| `FunctionSpace(mesh, "CG", 2)` | |
| `VectorFunctionSpace(mesh, "CG", 2)`, `TensorFunctionSpace(...)` | |
| `V * Q`, `MixedFunctionSpace([V, Q, R])` | mixed spaces |
| `FunctionSpace(mesh, element)` | with `FiniteElement("CG", triangle, 2)`, `VectorElement`, `TensorElement`, `MixedElement`, `E1 * E2` |
| `W.sub(0)`, `V.sub(1)`, `W.subspaces`, `len(W)` | component views for boundary conditions |
| `V.dim()`, `V.mesh()` | |
| `V.fes` | extra: the NGSolve space |

Families: `CG` (`Lagrange`, `P`, `Q`), `DG` (`DP`, `DQ`), `CR`, `R` (`Real`), `N1curl`,
`N2curl`, `RT`, `BDM`, `Regge`, `HHJ`.  Degrees follow FEniCS and Firedrake.  Extra
keyword arguments are passed to the NGSolve space.

## Functions

| | |
|---|---|
| `u = Function(V, name="u")` | extra: `u.gf` is the NGSolve `GridFunction`, `u.cf` its `CoefficientFunction` |
| `u.interpolate(expr)`, `u.project(expr)`, `u.assign(w)`, `u.zero()`, `u.copy(deepcopy=True)` | `expr`: UFL expression, number, tuple; extras: `lambda x: np.sin(x[0])` or an NGSolve CF |
| `u.at(point)`, `u.at(points)`, `u(point)` | point values |
| `u.subfunctions`, `u.sub(i)` | components of a mixed function, sharing its data |
| `u.dat.data`, `u.dat.data_ro`, `u.vector()` | coefficient array, see the note below |
| `u.name()`, `u.rename(name)`, `u.function_space()` | |
| `Constant(value)` | `c.assign(v)`, `c.value` |
| `from_cf(cf, mesh, degree=1)` | extra: an NGSolve CoefficientFunction as a UFL coefficient |
| `material_function(mesh, {"iron": 1000, "air": 1})` | extra: piecewise constant by material name |
| `to_cf(expr)` | extra: the NGSolve CoefficientFunction of a UFL expression |

```{note}
The coefficients belong to NGSolve's hierarchical basis.  `u.dat.data` holds point
values only for degree 1 Lagrange and degree 0 spaces.  For values use `u.at(...)`,
for manipulation `interpolate`, `project` and `assign`.
```

## Boundary conditions

`DirichletBC(V, g, sub_domain)` with `sub_domain` a boundary number, a tuple of numbers
or `"on_boundary"`.  `V` may be `W.sub(i)` or a component `V.sub(i)`; `g` is a number,
tuple, `Constant`, UFL expression or `Function`.  Extras: a name or regular expression,
or a callable, vectorised `lambda x: np.isclose(x[0], 0)` or `lambda x, on_boundary: ...`.
A callable that marks single points pins vertex values.

## Solving

| | |
|---|---|
| `solve(a == L, u, bcs=bcs, solver_parameters={...}, nullspace=...)` | linear problem |
| `solve(F == 0, u, bcs=bcs, J=None, ...)` | Newton; without `J` NGSolve differentiates `F` |
| `LinearVariationalProblem(a, L, u, bcs=None, constant_jacobian=False)`, `LinearVariationalSolver(problem, solver_parameters=..., nullspace=...)` | `constant_jacobian=True` factorises once |
| `NonlinearVariationalProblem(F, u, bcs=None, J=None)`, `NonlinearVariationalSolver(problem, ...)` | `.solve()`, `.iterations`; raises `ConvergenceError` |
| `assemble(form, bcs=None)` | number, `Vector` (alias `Cofunction`) or `Matrix`; extras: `.vec` and `.mat` are the NGSolve objects |
| `solve(A, x, b)`, `LinearSolver(A, solver_parameters=...).solve(x, b)` | assembled systems, honouring the `bcs` of `A` |
| `VectorSpaceBasis(constant=True)`, `VectorSpaceBasis([f1, f2])`, `MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])` | null spaces |
| `errornorm(u_exact, uh, "L2")`, `norm(expr, "H1")` | `L2`, `Lp`, `H1`, `H10`, `Hdiv`, `Hcurl` |
| `project(expr, V)`, `interpolate(expr, V)` | |

Solver options use PETSc names: `ksp_type` (`preonly`, `cg`, `gmres`, `minres`), `pc_type`
(`lu`, `cholesky`, `jacobi`, `none`), `ksp_rtol`, `ksp_max_it`, `ksp_monitor`, `snes_atol`,
`snes_rtol`, `snes_max_it`, `snes_monitor`, `snes_damping`.  Other Firedrake keywords
(`form_compiler_parameters`, `options_prefix`, `near_nullspace`, `appctx`, ...) are
accepted and ignored.

A null space is handled by fixing one coefficient per null space vector and removing
the null space component from the solution.

The quadrature order follows UFL's degree estimation.  Set it with `dx(degree=4)`.

## Output

| | |
|---|---|
| `from ngsufl.pyplot import tripcolor, tricontour, tricontourf, trisurf, triplot, quiver, plot` | the functions of `firedrake.pyplot`; they take `axes=` and return matplotlib artists |
| `VTKFile("name.pvd").write(u, p, time=t)` | VTK files |
| `plot(u)`, `plot(expr)`, `plot(mesh)` | extra: one-line matplotlib plot with colour bar; 3D boundary surface with optional `deformation=` |
| `Draw(u)` | extra: NGSolve webgui |

## Legacy FEniCS dialect

`from ngsufl.fenics import *` exports the same package with the names and call
signatures of dolfin 2019.  The differences to the default:

| | |
|---|---|
| `UnitSquareMesh(nx, ny, diagonal="right")`, `RectangleMesh(Point(x0, y0), Point(x1, y1), nx, ny)`, `BoxMesh(Point(...), Point(...), nx, ny, nz)`, `IntervalMesh(n, a, b)` | no boundary numbers; `mesh.coordinates()` is the vertex array |
| `Expression("1 + x[0]*x[0] + t", degree=2, t=0.0)` | C++ strings, parameters by attribute: `expr.t = 0.5` |
| `DirichletBC(V, g, boundary)` | `boundary`: function `(x, on_boundary)`, `"on_boundary"`, a C++ condition string, a `SubDomain` or `CompiledSubDomain` |
| `solve(a == L, u, bc)`, `solve(F == 0, u, bcs, J=J)` | `solver_parameters={"linear_solver": "cg", "preconditioner": "jacobi"}`, `{"newton_solver": {...}}` |
| `A = assemble(a)`, `b = assemble(L)`, `bc.apply(A, b)`, `solve(A, u.vector(), b)`, `assemble_system` | |
| `u.vector().get_local()`, `u.compute_vertex_values(mesh)` | the vertex values are true point values |
| `plot(u)`, `plot(mesh)`, `File("out.pvd") << u` | |
| `near`, `between`, `DOLFIN_EPS`, `Point`, `parameters`, `set_log_level` | |

Not available: `MeshFunction`, `SubDomain.mark`, `UserExpression`, `XDMFFile`, `mshr`.
See [the legacy dialect tutorial](tutorials/legacy_fenics).

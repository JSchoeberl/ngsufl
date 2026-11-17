# Status and limitations

ngsufl is experimental.  This page lists what is known not to work.

## Different by design

* **Dofs are not nodal.**  NGSolve's bases are hierarchical.  Use `u.at(x, y)`,
  `interpolate` and `project`, not the coefficient vector, to get at function values.
* **`interpolate` is a projection.**  It performs local $L^2$ projections followed by
  averaging, not nodal interpolation.
* **Boundaries have names.**  Subdomains are addressed by their NGSolve names or by
  1-based numbers.  There are no mesh tags.

## Not there yet

* complex arithmetic
* `dS` restricted to a subset of the interior facets
* symmetric and non-square tensor valued elements, enriched and bubble elements,
  quadrature elements, nested mixed spaces
* `FacetArea`; second derivatives of H(div), H(curl) and matrix valued functions
* mesh tags
* MPI
* algebraic multigrid and domain decomposition preconditioners through the solver
  options (the NGSolve preconditioners can be used directly on the assembled matrices)
* PETSc solvers; planned through [ngsPETSc](https://github.com/NGSolve/ngsPETSc)

## Things to know

* For residuals with interior facet terms the Jacobian always comes from UFL's
  `derivative`, since NGSolve does not linearise skeleton integrals.
* `Jacobian` refers to NGSolve's reference cells, whose vertex order differs from
  FEniCS.  The determinant agrees up to sign.
* `Constant(value)` without a mesh belongs to the mesh created last.
* A coefficient of one mesh inside an integral over another mesh is not detected.

## Tests

The test suite compares against numpy formulas and against forms written directly in
NGSolve: expressions, derivatives, assembled matrices, element dimensions, boundary
conditions, convergence rates.  Known gaps are kept as expected failures.

## Front end

The front end follows Firedrake and legacy FEniCS where they agree.  It is deliberately
small.  Which of the existing UFL front ends it should imitate more closely is an open
question.

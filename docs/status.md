# Status and limitations

ngsufl is experimental.  This page lists what is known not to work.

## Different by design

* **Dofs are not nodal.**  NGSolve's bases are hierarchical.  Use `u.at(x, y)`,
  `interpolate` and `project` to get at function values.  `u.dat.data` and
  `u.vector()` return the coefficients, which are point values only for degree 1
  Lagrange and degree 0 spaces.
* **`interpolate` is the canonical interpolation** of the NGSolve space: vertex values
  and edge, face and cell moments for Lagrange spaces, the commuting interpolation for
  H(curl) and H(div), a local $L^2$ projection for discontinuous spaces.  Vertex values
  are reproduced exactly, the values at the other Lagrange nodes are not.
* **Boundaries have names.**  The built-in meshes carry Firedrake's boundary numbers,
  and every boundary also has its NGSolve name.  Meshes from Netgen or OpenCascade
  geometries are addressed by name, or by the 1-based NGSolve index.  There are no
  mesh tags.
* **Null spaces** are handled by fixing one coefficient per null space vector and
  removing the null space component afterwards, not inside a Krylov method.

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

The default front end follows Firedrake, `ngsufl.fenics` follows legacy FEniCS
(dolfin 2019).  Both cover the common path of tutorial-style scripts, not the whole
of either library.  The DOLFINx interface is not imitated: it exposes dof arrays and
PETSc objects directly, which does not fit hierarchical bases.

Not available from Firedrake: periodic, extruded and immersed meshes, mesh
hierarchies and geometric multigrid, `fieldsplit` and the other PETSc
preconditioners, Slate, `Interpolator` and `VertexOnlyMesh`, checkpointing, adjoints.

Not available from legacy FEniCS: `MeshFunction` and `SubDomain.mark`,
`UserExpression`, `XDMFFile`, `mshr`, the PETSc and Eigen linear algebra back ends.

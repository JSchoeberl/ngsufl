# How it works

UFL is a symbolic language without a back end.  FFCx (DOLFINx) and TSFC (Firedrake)
compile UFL forms to element kernels.  ngsufl is a third back end, and it compiles
nothing: a UFL expression becomes an NGSolve `CoefficientFunction`, and NGSolve's
symbolic integrators do the rest.

## Expressions

A UFL expression tree is walked once.

| UFL | NGSolve |
|---|---|
| `TrialFunction`, `TestFunction` | trial and test proxies of the space |
| `Function` (a UFL `Coefficient`) | `GridFunction` |
| `Constant` | `Parameter` |
| `grad`, `div`, `curl` of a function | differential operators of the space |
| `inner`, `dot`, `outer`, `cross` | `InnerProduct`, `*`, `OuterProduct`, `Cross` |
| `sym`, `skew`, `dev`, `tr`, `det`, `inv`, `cofac` | `Sym`, `Skew`, `Deviator`, `Trace`, `Det`, `Inv`, `Cof` |
| `conditional`, `lt`, `gt`, `max_value` | `IfPos` |
| `SpatialCoordinate`, `FacetNormal` | `x, y, z`, `specialcf.normal` |
| `CellDiameter`, `CellVolume`, `Circumradius` | piecewise constant `GridFunction`, computed from the vertices |
| `u('+')`, `u('-')` | `u`, `u.Other()` |
| `dx`, `ds`, `dS` | `dx`, `ds(skeleton=True)`, `dx(skeleton=True)` |

**Index notation.**  UFL lowers many operations to index notation, for example every
product of a scalar with a tensor.  The translator collects products of indexed tensors
into lazy Einstein products.  When the product is closed by a sum or an `as_tensor`, it
is emitted as a native operation if a pattern is recognised: `s * A[i,j]` is a scalar
multiple, `A[i,k] * B[k,j]` a matrix product, `A[i,j] * B[i,j]` an inner product.
Everything else becomes one `Einsum`.

**Derivatives.**  `grad`, `div` and `curl` applied to a function, or to a component of a
mixed function obtained with `split`, map to the operators of the NGSolve space.
Applied to a general expression, UFL's own derivative expansion pushes them down to the
functions first.  The same is done for `derivative` and `diff`.

**No pullbacks.**  FFCx and TSFC rewrite a form on the reference cell.  NGSolve applies
the Piola transformations itself, so the translator stays in physical space.

## Elements

A FEniCS element name is a row in a table.

| UFL family, degree $k$ | NGSolve space |
|---|---|
| `Lagrange` | `H1(order=k)` |
| `DG` | `L2(order=k)` |
| `N1curl` | `HCurl(order=k, type1=True)` |
| `N2curl` | `HCurl(order=k)` |
| `RT` | `HDiv(order=k-1, RT=True)` |
| `BDM` | `HDiv(order=k)` |
| `Regge` | `HCurlCurl(order=k)` |
| `HHJ` | `HDivDiv(order=k)` |
| `CR` | `FESpace("nonconforming")` |
| `Real` | `NumberSpace` |

The spaces span the same polynomials as their FEniCS namesakes, with NGSolve's
hierarchical bases.  The coefficient vector therefore does not hold nodal values.

## Forms

* A form with two arguments becomes a `BilinearForm`, with one a `LinearForm`, with none
  a number.
* The quadrature order of each integral follows UFL's degree estimation, as in FEniCS,
  and can be set with `dx(degree=q)`.
* Forms with `dS` integrals are assembled on a twin of the space that carries the
  facet couplings in its sparsity pattern.

## Nonlinear problems

For `solve(F == 0, u)` the residual `F(u; v)` is turned into a nonlinear NGSolve
`BilinearForm` by putting the trial function in place of `u`.  NGSolve then
differentiates it symbolically for Newton's method.  If a Jacobian
`J = derivative(F, u)` is passed, UFL's derivative is assembled instead.  The two routes
agree to round-off.

## Boundary conditions

NGSolve usually declares Dirichlet boundaries when the space is created.  A
`DirichletBC` here is created afterwards, as in FEniCS: it marks the dofs of a boundary
region and the solver restricts the system to the remaining ones.

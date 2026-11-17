# %% [markdown]
# # Linear elasticity
#
# A clamped beam deforming under its own weight, following the linear elasticity
# chapter of the [FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/) and the
# elasticity demo among the [DOLFINx demos](https://docs.fenicsproject.org/dolfinx/main/python/demos.html).
# The displacement $u$ solves
#
# $$ -\nabla\cdot\sigma(u) = f \ \text{ in } \Omega, \qquad u = 0 \ \text{ on } \Gamma_D, \qquad \sigma(u)\,n = 0 \ \text{ on } \partial\Omega\setminus\Gamma_D, $$
#
# with the strain $\varepsilon(u) = \tfrac12(\nabla u + \nabla u^T)$ and the stress
# $\sigma(u) = \lambda\,\mathrm{tr}(\varepsilon)\,I + 2\mu\,\varepsilon$.
# The beam occupies $(0,L)\times(0,W)\times(0,W)$, is clamped at $x=0$ and loaded by gravity
# $f = (0, 0, -\rho g)$.  The weak form is: find $u \in V_0$ with
#
# $$ \int_\Omega \sigma(u) : \varepsilon(v) \, dx = \int_\Omega f \cdot v \, dx \qquad \forall v \in V_0 . $$

# %% [markdown]
# ## Mesh, space and material
#
# `BoxMesh` creates a tetrahedral mesh; its faces are named `back` ($x=0$), `front`
# ($x=L$), `left` ($y=0$), `right` ($y=W$), `bottom` ($z=0$) and `top` ($z=W$).

# %%
from ngsufl import *

L, W = 1.0, 0.2
mesh = BoxMesh(10, 3, 3, L, W, W)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
print(mesh.boundaries, V.dim(), "dofs")

delta = W / L
gamma = 0.4 * delta**2
rho, g = 1.0, gamma
mu, lmbda = 1.0, 1.25

# %% [markdown]
# ## Variational problem
#
# The stress is a symmetric $3\times3$ matrix expression, written in the same way as on paper.

# %%
def epsilon(u):
    return sym(grad(u))


def sigma(u):
    return lmbda * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)


u, v = TrialFunction(V), TestFunction(V)
f = Constant((0.0, 0.0, -rho * g))

a = inner(sigma(u), epsilon(v)) * dx
L_form = dot(f, v) * dx

# %% [markdown]
# ## Clamping and solution
#
# The Dirichlet condition is imposed on the boundary named `back`.  Direct solvers are the
# default; for large 3D problems one would select `cg` with a preconditioner through
# `solver_parameters`.

# %%
bc = DirichletBC(V, (0.0, 0.0, 0.0), "back")
uh = Function(V, name="u")
solve(a == L_form, uh, bc)

tip = uh.at(L, W / 2, W / 2)
print("tip displacement:", tip)

# %% [markdown]
# ## Deformed shape
#
# `plot` draws the boundary surface of a 3D mesh coloured by the given function.  With
# `deformation=` the surface is moved by a vector field, here the
# displacement itself (`deformation=10 * uh` would exaggerate it).

# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(projection="3d")
plot(sqrt(dot(uh, uh)), ax=ax, deformation=uh, subdivision=2)
ax.set_aspect("equal")
ax.set_title("displacement magnitude on the deformed beam");

# %% [markdown]
# ## Von Mises stress
#
# The von Mises stress $\sigma_M = \sqrt{\tfrac32\, s:s}$ with the deviator
# $s = \sigma - \tfrac13\mathrm{tr}(\sigma) I$ is a derived quantity.  We project it onto a
# discontinuous space and plot it.

# %%
s = sigma(uh) - tr(sigma(uh)) / 3 * Identity(3)
von_Mises = sqrt(1.5 * inner(s, s))

Q = FunctionSpace(mesh, "DG", 1)
stress = project(von_Mises, Q)
stress.name = "von Mises"
print("max von Mises stress:", stress.array.max())

# %%
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(projection="3d")
plot(stress, ax=ax, deformation=uh, subdivision=2)
ax.set_aspect("equal")
ax.set_title("von Mises stress");

# %% [markdown]
# ## Comparison with plain NGSolve
#
# The same problem written directly with NGSolve objects should give the same
# discrete solution up to round-off, since both use the same space and the same
# exact quadrature.  UFL is only a different front door to the NGSolve integrators.

# %%
import ngsolve as ngs

m = mesh.ngs
fes = ngs.VectorH1(m, order=2, dirichlet="back")
U, Wt = fes.TnT()
E = lambda w: ngs.Sym(ngs.Grad(w))
S = lambda w: lmbda * ngs.Trace(E(w)) * ngs.Id(3) + 2 * mu * E(w)

A = ngs.BilinearForm(ngs.InnerProduct(S(U), E(Wt)) * ngs.dx).Assemble()
F = ngs.LinearForm(ngs.CF((0, 0, -rho * g)) * Wt * ngs.dx).Assemble()
gf = ngs.GridFunction(fes)
gf.vec.data = A.mat.Inverse(fes.FreeDofs()) * F.vec

tip_ngs = gf(m(L, W / 2, W / 2))
print("native NGSolve tip:", tip_ngs)
print("difference of tip values:", max(abs(a_ - b_) for a_, b_ in zip(tip, tip_ngs)))
print("L2 difference of fields:", ngs.sqrt(ngs.Integrate((uh.gf - gf)**2, m)))

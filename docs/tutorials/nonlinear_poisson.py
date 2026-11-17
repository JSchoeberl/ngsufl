# %% [markdown]
# # A nonlinear Poisson equation
#
# This tutorial follows the chapter on nonlinear problems in the
# [FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/).
# We solve
#
# $$ -\nabla\cdot\big(q(u)\nabla u\big) = f \quad \text{in } (0,1)^2, \qquad u = u_D \quad \text{on } \partial\Omega, \qquad q(u) = 1 + u^2 . $$
#
# The manufactured solution is $u_e = 1 + x + 2y$.  UFL differentiates symbolically, so the
# source term $f = -\nabla\cdot(q(u_e)\nabla u_e)$ is obtained by writing exactly that.
# Because $u_e$ is linear in $x$ and $y$, first order elements can in principle represent
# it; but the nonlinearity makes the discrete problem nonlinear, and we have to find the
# root of the residual by Newton's method.

# %% [markdown]
# ## Residual form
#
# Now the unknown `u` is a `Function` (not a `TrialFunction`), and the weak form is
# written as a residual $F(u; v) = 0$.

# %%
from ngsufl import *

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "Lagrange", 1)
x = SpatialCoordinate(mesh)


def q(u):
    return 1 + u**2


u_e = 1 + x[0] + 2 * x[1]
f = -div(q(u_e) * grad(u_e))

v = TestFunction(V)
bc = DirichletBC(V, u_e, "on_boundary")

# %% [markdown]
# ## Newton's method, first variant: no Jacobian given
#
# If we do not pass a Jacobian, NGSolve linearises the residual by itself.  We switch on
# the monitor of the nonlinear solver to see the residual norm in each iteration.

# %%
u1 = Function(V, name="u_ad")
F1 = q(u1) * inner(grad(u1), grad(v)) * dx - f * v * dx

solver1 = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F1, u1, bc),
    {"snes_monitor": True, "snes_rtol": 1e-10},
)
solver1.solve()
print("Newton iterations:", solver1.iterations)

# %% [markdown]
# ## Second variant: the Jacobian from UFL
#
# This is what a FEniCS or Firedrake script does: `derivative(F, u)` produces the
# linearised form symbolically, and we hand it to the solver as `J`.

# %%
u2 = Function(V, name="u_ufl")
F2 = q(u2) * inner(grad(u2), grad(v)) * dx - f * v * dx
J2 = derivative(F2, u2)

solver2 = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F2, u2, bc, J=J2),
    {"snes_monitor": True, "snes_rtol": 1e-10},
)
solver2.solve()
print("Newton iterations:", solver2.iterations)

# %% [markdown]
# ## Comparison
#
# Both runs should need the same number of iterations and deliver the same solution.
# We also compare with the exact solution; the remaining error is the discretisation
# error of the first order elements.

# %%
print("difference between the two solutions (L2):", errornorm(u1, u2))
print("error vs exact, AD  :", errornorm(u_e, u1))
print("error vs exact, UFL :", errornorm(u_e, u2))

# %% [markdown]
# ## The Jacobian is really used
#
# The two runs agree to the last digit, which is expected: both Jacobians are the exact
# derivative of the same residual.  That the form we pass is the one used becomes visible
# when we give a different one.  The Picard linearisation freezes the coefficient,
# $J = \int q(u)\,\nabla\delta u\cdot\nabla v\,dx$, and converges only linearly.

# %%
u3 = Function(V, name="u_picard")
F3 = q(u3) * inner(grad(u3), grad(v)) * dx - f * v * dx
J3 = q(u3) * inner(grad(TrialFunction(V)), grad(v)) * dx

solver3 = NonlinearVariationalSolver(NonlinearVariationalProblem(F3, u3, bc, J=J3), {"snes_rtol": 1e-10})
solver3.solve()
print("Picard iterations:", solver3.iterations, " error vs exact:", errornorm(u_e, u3))

# %% [markdown]
# ## Order of convergence
#
# A quick check that the nonlinear solve is consistent: with quadratic elements and
# successively finer meshes the $L^2$ error drops by a factor close to 8 per halving of $h$.
# The exact solution is not in $P_2$ any more if we choose a non-polynomial one, so we
# use $u_e = 1 + \sin(\pi x)\,y$ here.

# %%
import numpy as np


def solve_for(n):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "Lagrange", 2)
    x = SpatialCoordinate(mesh)
    ue = 1 + sin(pi * x[0]) * x[1]
    f = -div(q(ue) * grad(ue))
    u, v = Function(V), TestFunction(V)
    F = q(u) * inner(grad(u), grad(v)) * dx - f * v * dx
    solve(F == 0, u, DirichletBC(V, ue, "on_boundary"), J=derivative(F, u))
    return errornorm(ue, u)


errs = [solve_for(n) for n in (4, 8, 16)]
print("errors:", errs)
print("observed rates:", [round(float(np.log2(errs[i] / errs[i + 1])), 2) for i in range(2)])

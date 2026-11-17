"""assemble, solve and the variational problem / solver classes."""

import math
import numbers

import ngsolve as ngs
import ufl
from ufl.equation import Equation

from . import forms
from .bcs import DirichletBC, free_dofs
from .function import Function, as_cf

_DIRECT = {"lu": "", "cholesky": "sparsecholesky", "umfpack": "umfpack",
           "pardiso": "pardiso", "sparsecholesky": "sparsecholesky", "mumps": "mumps"}
_KRYLOV = {"cg": ngs.krylovspace.CGSolver, "gmres": ngs.krylovspace.GMResSolver,
           "minres": ngs.krylovspace.MinResSolver}


def _bcs(bcs):
    if bcs is None:
        return []
    return [bcs] if isinstance(bcs, DirichletBC) else list(bcs)


def _is_zero(L):
    return L is None or (isinstance(L, numbers.Number) and L == 0) or \
        (isinstance(L, ufl.Form) and L.empty()) or isinstance(L, ufl.classes.Zero)


def assemble(form, **flags):
    """Number, NGSolve vector or NGSolve matrix, depending on the arity of the form."""
    arity = len(form.arguments())
    if arity == 0:
        return forms.functional(form)
    if arity == 1:
        return forms.linear_form(form).Assemble().vec
    return forms.bilinear_form(form, **flags).Assemble().mat


def make_inverse(mat, free, params=None):
    """Solver for the free dofs, chosen by PETSc style options.

    ksp_type: preonly (default), cg, gmres, minres
    pc_type : lu (default), cholesky, umfpack, pardiso, jacobi, none
    """
    p = dict(params or {})
    ksp, pc = p.get("ksp_type", "preonly"), p.get("pc_type", "lu")
    if ksp == "preonly":
        if pc not in _DIRECT:
            raise NotImplementedError(f"pc_type '{pc}' with ksp_type 'preonly'")
        return mat.Inverse(free, inverse=_DIRECT[pc])
    if ksp not in _KRYLOV:
        raise NotImplementedError(f"ksp_type '{ksp}'; for the full PETSc range use ngsPETSc")
    if pc == "jacobi":
        pre = mat.CreateSmoother(free)
    elif pc == "none":
        pre = ngs.Projector(free, True)
    elif pc in _DIRECT:
        pre = mat.Inverse(free, inverse=_DIRECT[pc])
    else:
        raise NotImplementedError(f"pc_type '{pc}'; for the full PETSc range use ngsPETSc")
    proj = ngs.Projector(free, True)
    solver = _KRYLOV[ksp](mat=proj @ mat @ proj, pre=pre, tol=p.get("ksp_rtol", 1e-10),
                          maxiter=p.get("ksp_max_it", 1000), printrates=bool(p.get("ksp_monitor", False)))
    return solver @ proj


class LinearVariationalProblem:
    def __init__(self, a, L, u, bcs=None, constant_jacobian=False):
        self.a, self.L, self.u, self.bcs = a, L, u, _bcs(bcs)
        self.constant_jacobian = constant_jacobian


class LinearVariationalSolver:
    def __init__(self, problem, solver_parameters=None, **flags):
        self.problem, self.params, self.flags = problem, solver_parameters, flags
        self._a = self._inv = self._f = self._free = None

    def solve(self):
        p = self.problem
        gf = p.u.gf
        if self._free is None:
            self._free = free_dofs(p.u.function_space(), p.bcs)
        if self._a is None:
            self._a = forms.bilinear_form(p.a, **self.flags)
        if self._inv is None or not p.constant_jacobian:
            self._a.Assemble()
            self._inv = make_inverse(self._a.mat, self._free, self.params)
        for bc in p.bcs:
            bc.apply(gf)
        res = self._a.mat.CreateColVector()
        if _is_zero(p.L):
            res[:] = 0.0
        else:
            if self._f is None:
                self._f = forms.linear_form(p.L)
            res.data = self._f.Assemble().vec
        res.data -= self._a.mat * gf.vec
        gf.vec.data += self._inv * res


class NonlinearVariationalProblem:
    def __init__(self, F, u, bcs=None, J=None):
        self.F, self.u, self.bcs, self.J = F, u, _bcs(bcs), J


class NonlinearVariationalSolver:
    """Newton's method.  Without J the tangent comes from NGSolve's own differentiation.

    Options: snes_atol (1e-10), snes_rtol (1e-9), snes_max_it (50), snes_monitor,
    snes_damping (1.0), plus the linear solver options.
    """

    def __init__(self, problem, solver_parameters=None, **flags):
        self.problem, self.params, self.flags = problem, dict(solver_parameters or {}), flags
        self._res = self._jac = self._free = None
        self.iterations = 0

    def solve(self):
        p, par = self.problem, self.params
        gf = p.u.gf
        if self._free is None:
            self._free = free_dofs(p.u.function_space(), p.bcs)
            self._res = forms.residual_form(p.F, p.u, **self.flags)
            J = p.J
            if J is None and forms.has_interior_facets(p.F):
                # NGSolve does not linearise interior facet terms
                J = ufl.derivative(p.F, p.u)
            if J is not None:
                self._jac = forms.bilinear_form(J, **self.flags)
        for bc in p.bcs:
            bc.apply(gf)
        atol, rtol = par.get("snes_atol", 1e-10), par.get("snes_rtol", 1e-9)
        maxit, damp = par.get("snes_max_it", 50), par.get("snes_damping", 1.0)
        res = gf.vec.CreateVector()
        du = gf.vec.CreateVector()
        proj = ngs.Projector(self._free, True)
        r0 = None
        for it in range(maxit + 1):
            self._res.Apply(gf.vec, res)
            proj.Project(res)
            r = ngs.Norm(res)
            r0 = r if r0 is None else r0
            if par.get("snes_monitor"):
                print(f"Newton iteration {it}: residual = {r:.3e}")
            if r < atol or r < rtol * r0:
                self.iterations = it
                return
            if not math.isfinite(r) or it == maxit:
                break
            if self._jac is not None:
                self._jac.Assemble()
                mat = self._jac.mat
            else:
                self._res.AssembleLinearization(gf.vec)
                mat = self._res.mat
            du.data = make_inverse(mat, self._free, par) * res
            gf.vec.data -= damp * du
        raise RuntimeError(f"Newton did not converge, residual {r:.3e} after {it} iterations")


def solve(equation, u, bcs=None, J=None, solver_parameters=None, **flags):
    """``solve(a == L, u, bcs)`` or ``solve(F == 0, u, bcs)``."""
    if not isinstance(equation, Equation):
        raise TypeError("expected an equation a == L or F == 0")
    lhs, rhs = equation.lhs, equation.rhs
    if len(lhs.arguments()) == 2:
        prob = LinearVariationalProblem(lhs, rhs, u, bcs)
        LinearVariationalSolver(prob, solver_parameters, **flags).solve()
        return
    if not _is_zero(rhs):
        lhs = lhs - rhs
    prob = NonlinearVariationalProblem(lhs, u, bcs, J)
    NonlinearVariationalSolver(prob, solver_parameters, **flags).solve()


def project(expr, V, bcs=None, solver_parameters=None, name=None):
    """Global L2 projection."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    w = Function(V, name=name)
    if not isinstance(expr, ufl.core.expr.Expr):
        expr = Function(V).interpolate(expr)
    solve(ufl.inner(u, v) * ufl.dx == ufl.inner(expr, v) * ufl.dx, w, bcs, solver_parameters=solver_parameters)
    return w


def interpolate(expr, V, name=None):
    return Function(V, name=name).interpolate(expr)


def norm(u, norm_type="L2"):
    """L2, H1 (full), H10 (seminorm), Hdiv or Hcurl norm of an expression."""
    t = norm_type.lower()
    e = ufl.inner(u, u)
    if t == "h1":
        e = e + ufl.inner(ufl.grad(u), ufl.grad(u))
    elif t == "h10":
        e = ufl.inner(ufl.grad(u), ufl.grad(u))
    elif t == "hdiv":
        e = e + ufl.inner(ufl.div(u), ufl.div(u))
    elif t == "hcurl":
        e = e + ufl.inner(ufl.curl(u), ufl.curl(u))
    elif t != "l2":
        raise ValueError(f"unknown norm '{norm_type}'")
    return math.sqrt(abs(assemble(e * ufl.dx)))


def errornorm(u_exact, uh, norm_type="L2"):
    return norm(u_exact - uh, norm_type)

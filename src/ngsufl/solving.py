"""assemble, solve and the variational problem / solver classes."""

import math
import numbers

import ngsolve as ngs
import numpy as np
import ufl
from ufl.equation import Equation

from . import forms
from .bcs import DirichletBC, free_dofs
from .function import Function, Vector, as_cf

# Firedrake keywords without a counterpart here
_IGNORED = {"form_compiler_parameters", "options_prefix", "appctx", "mat_type", "sub_mat_type",
            "near_nullspace", "transpose_nullspace", "pre_jacobian_callback", "post_jacobian_callback",
            "pre_function_callback", "post_function_callback", "restrict", "aP", "Jp", "Jp_eq_J",
            "zero_bc_nodes", "pre_apply_bcs", "tensor", "weight", "is_linear", "P"}


def _flags(kw):
    return {k: v for k, v in kw.items() if k not in _IGNORED}


class ConvergenceError(RuntimeError):
    pass

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


class Matrix:
    """An assembled bilinear form: ``.mat`` is the NGSolve matrix, ``.bcs`` its boundary conditions."""

    def __init__(self, a, mat, space, bcs=()):
        self.a, self.mat, self._space, self.bcs = a, mat, space, list(bcs)

    def __getattr__(self, name):
        if name == "mat":
            raise AttributeError(name)
        return getattr(self.mat, name)

    @property
    def M(self):
        return self

    @property
    def handle(self):
        raise NotImplementedError("PETSc matrices need the ngsPETSc route; the NGSolve matrix is .mat")

    petscmat = handle

    def array(self):
        """Dense matrix, with unit rows for the boundary conditions."""
        A = np.array(self.mat.ToDense().NumPy())
        for bc in self.bcs:
            idx = np.nonzero(np.array(list(bc.dofs()), dtype=bool))[0]
            A[idx, :] = 0.0
            A[idx, idx] = 1.0
        return A

    def __mul__(self, x):
        if isinstance(x, Vector):
            y = Vector(self.mat.CreateColVector(), self._space)
            y.vec.data = self.mat * x.vec
            return y
        return self.mat * x

    __matmul__ = __mul__


def assemble(expr, bcs=None, **kw):
    """Number, `Vector` or `Matrix`, depending on the arity of the form."""
    if not isinstance(expr, ufl.Form):
        return expr
    arity = len(expr.arguments())
    if arity == 0:
        return forms.functional(expr)
    V = expr.arguments()[0].ufl_function_space()
    if arity == 1:
        b = Vector(forms.linear_form(expr).Assemble().vec, V)
        for bc in _bcs(bcs):
            bc.apply(b)
        return b
    return Matrix(expr, forms.bilinear_form(expr, **_flags(kw)).Assemble().mat, V, _bcs(bcs))


class VectorSpaceBasis:
    """Null space of an operator: given functions, and the constants with ``constant=True``."""

    def __init__(self, vecs=None, constant=False, comm=None):
        self.vecs, self.constant = list(vecs or []), constant

    def orthonormalize(self):
        pass

    def _vectors(self, V):
        n, rng, out = V.root.fes.ndof, V.dof_range(), []
        if self.constant:
            gf = ngs.GridFunction(V.fes)
            gf.Set(as_cf(1.0, V.mesh, tuple(gf.dims) if gf.dim > 1 else ()))
            out.append(gf.vec.FV().NumPy().copy())
        out += [f.gf.vec.FV().NumPy().copy() for f in self.vecs]
        full = []
        for v in out:
            w = np.zeros(n)
            w[rng.start:rng.stop] = v
            full.append(w)
        return full


class MixedVectorSpaceBasis:
    """``MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])``."""

    def __init__(self, function_space, bases):
        self.space, self.bases = function_space, list(bases)

    def orthonormalize(self):
        pass

    def _vectors(self, V):
        out = []
        for i, b in enumerate(self.bases):
            if isinstance(b, VectorSpaceBasis):
                out += b._vectors(self.space.sub(i))
        return out


def _constrain(free, nullspace, V):
    """Fix one dof per null space vector; returns the free dofs and an orthonormal basis."""
    if nullspace is None:
        return free, None
    N = np.array(nullspace._vectors(V))
    if not len(N):
        return free, None
    free = ngs.BitArray(free)
    R = N * np.array(list(free), dtype=bool)
    for i in range(len(R)):
        j = int(np.argmax(np.abs(R[i])))
        free.Clear(j)
        for k in range(len(R)):
            if k != i:
                R[k] -= R[k, j] / R[i, j] * R[i]
        R[:, j] = 0.0
    return free, np.linalg.qr(N.T)[0]


def _remove(vec, basis):
    if basis is not None:
        x = vec.FV().NumPy()
        x -= basis @ (basis.T @ x)


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


class LinearSolver:
    """``LinearSolver(A).solve(x, b)`` for an assembled `Matrix`, honouring its boundary conditions."""

    def __init__(self, A, solver_parameters=None, nullspace=None, **kw):
        self.A, self.params = A, solver_parameters
        V = A._space
        self._free, self._ns = _constrain(free_dofs(V, A.bcs), nullspace, V)
        self._inv = None

    def solve(self, x, b):
        A = self.A
        xv = x.gf.vec if isinstance(x, Function) else getattr(x, "vec", x)
        bv = b.gf.vec if isinstance(b, Function) else getattr(b, "vec", b)
        if A.bcs:
            gf = ngs.GridFunction(A._space.fes)
            gf.vec.data = xv
            for bc in A.bcs:
                bc.apply(gf)
            xv.data = gf.vec
        if self._inv is None:
            self._inv = make_inverse(A.mat, self._free, self.params)
        res = bv.CreateVector()
        res.data = bv - A.mat * xv
        xv.data += self._inv * res
        _remove(xv, self._ns)


class LinearVariationalProblem:
    def __init__(self, a, L, u, bcs=None, constant_jacobian=False, **kw):
        self.a, self.L, self.u, self.bcs = a, L, u, _bcs(bcs)
        self.constant_jacobian = constant_jacobian


class LinearVariationalSolver:
    def __init__(self, problem, solver_parameters=None, nullspace=None, **kw):
        self.problem, self.params, self.flags = problem, solver_parameters, _flags(kw)
        self.nullspace = nullspace
        self._a = self._inv = self._f = self._free = self._ns = None

    def solve(self):
        p = self.problem
        gf = p.u.gf
        if self._free is None:
            V = p.u.function_space()
            self._free, self._ns = _constrain(free_dofs(V, p.bcs), self.nullspace, V)
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
        _remove(gf.vec, self._ns)


class NonlinearVariationalProblem:
    def __init__(self, F, u, bcs=None, J=None, **kw):
        self.F, self.u, self.bcs, self.J = F, u, _bcs(bcs), J


class NonlinearVariationalSolver:
    """Newton's method.  Without J the tangent comes from NGSolve's own differentiation.

    Options: snes_atol (1e-10), snes_rtol (1e-9), snes_max_it (50), snes_monitor,
    snes_damping (1.0), plus the linear solver options.
    """

    def __init__(self, problem, solver_parameters=None, nullspace=None, **kw):
        self.problem, self.params, self.flags = problem, dict(solver_parameters or {}), _flags(kw)
        self.nullspace = nullspace
        self._res = self._jac = self._free = self._ns = None
        self.iterations = 0

    def solve(self):
        p, par = self.problem, self.params
        gf = p.u.gf
        if self._free is None:
            V = p.u.function_space()
            self._free, self._ns = _constrain(free_dofs(V, p.bcs), self.nullspace, V)
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
                _remove(gf.vec, self._ns)
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
        raise ConvergenceError(f"Newton did not converge, residual {r:.3e} after {it} iterations")


def solve(equation, u, bcs=None, J=None, solver_parameters=None, nullspace=None, **kw):
    """``solve(a == L, u, bcs)``, ``solve(F == 0, u, bcs)`` or ``solve(A, x, b)`` for assembled A, b."""
    if isinstance(equation, Matrix):
        LinearSolver(equation, solver_parameters, nullspace).solve(u, bcs)
        return
    if not isinstance(equation, Equation):
        raise TypeError("expected an equation a == L or F == 0, or an assembled matrix")
    lhs, rhs = equation.lhs, equation.rhs
    if len(lhs.arguments()) == 2:
        prob = LinearVariationalProblem(lhs, rhs, u, bcs)
        LinearVariationalSolver(prob, solver_parameters, nullspace, **kw).solve()
        return
    if not _is_zero(rhs):
        lhs = lhs - rhs
    prob = NonlinearVariationalProblem(lhs, u, bcs, J)
    NonlinearVariationalSolver(prob, solver_parameters, nullspace, **kw).solve()


def project(expr, V, bcs=None, solver_parameters=None, name=None, **kw):
    """Global L2 projection."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    w = Function(V, name=name)
    if not isinstance(expr, ufl.core.expr.Expr):
        expr = Function(V).interpolate(expr)
    solve(ufl.inner(u, v) * ufl.dx == ufl.inner(expr, v) * ufl.dx, w, bcs, solver_parameters=solver_parameters)
    return w


def interpolate(expr, V, name=None, **kw):
    """Interpolate into V; ``assemble(interpolate(expr, V))`` works as well."""
    return Function(V, name=name).interpolate(expr)


def norm(u, norm_type="L2", mesh=None):
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
    elif t.startswith("l") and t[1:].isdigit() and t != "l2":
        p = int(t[1:])
        return abs(assemble(e ** (p / 2) * ufl.dx)) ** (1 / p)
    elif t != "l2":
        raise ValueError(f"unknown norm '{norm_type}'")
    return math.sqrt(abs(assemble(e * ufl.dx)))


def errornorm(u_exact, uh, norm_type="L2", **kw):
    return norm(u_exact - uh, norm_type)

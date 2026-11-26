"""Adjoints and derivatives of functionals with pyadjoint (prototype).

    from ngsufl import *
    from ngsufl.pyadjoint import *

    continue_annotation()
    solve(F == 0, u, bcs=bc)
    J = assemble(inner(u - ud, u - ud) * dx)
    Jhat = ReducedFunctional(J, Control(m))

Importing this module makes `Function` and `Constant` pyadjoint types and records
`solve`, the variational solvers, `project`, `assemble` of functionals and
`Function.assign` / `interpolate` / `copy` on the tape.  Needs ``pyadjoint-ad``.

Covered: first derivatives with respect to Functions and Constants.  Not covered:
Hessians and tangent linear models, Dirichlet values as controls, assembled matrices
and vectors, components of mixed functions (``u.subfunctions``; use ``split(u)``),
shape derivatives.  Time stepping with Irksome is therefore not recorded either.
"""

import numbers

import numpy as np
import ufl
from ufl.algorithms.analysis import extract_coefficients, extract_type

from pyadjoint import (AdjFloat, Block, Control, ReducedFunctional, Tape, annotate_tape,  # noqa: F401
                       compute_gradient, continue_annotation, get_working_tape, maximize, minimize,
                       pause_annotation, set_working_tape, stop_annotating, taylor_test)
from pyadjoint.overloaded_type import OverloadedType, Weakref

import ngsufl
from ngsufl import solving as _sv
from ngsufl.forms import mesh_of
from ngsufl.function import DomainConstant, Function, FunctionSpace, GlobalConstant

__all__ = ["AdjFloat", "Control", "ReducedFunctional", "Tape", "annotate_tape", "assemble",
           "compute_gradient", "continue_annotation", "get_working_tape", "maximize", "minimize",
           "pause_annotation", "set_working_tape", "stop_annotating", "taylor_test"]

_CONSTANTS = (GlobalConstant, DomainConstant)


# ---- values

class Dual:
    """Derivative with respect to a Function: one number per coefficient."""

    def __init__(self, array, space):
        self.array, self.space = np.array(array, dtype=float), space

    def function_space(self):
        return self.space

    def riesz_representation(self, riesz_map="L2"):
        return _riesz(self, riesz_map)

    def __add__(self, other):
        return Dual(self.array + other.array, self.space)

    __iadd__ = __add__

    def __mul__(self, a):
        return Dual(self.array * float(a), self.space)

    __rmul__ = __mul__

    def __neg__(self):
        return Dual(-self.array, self.space)

    def _ad_dot(self, other):
        return float(self.array @ _coefficients(other))


def _coefficients(f):
    return f.array if isinstance(f, Dual) else f.gf.vec.FV().NumPy()


def _copy(f):
    g = Function(f.function_space(), name=f.name)
    g.gf.vec.data = f.gf.vec
    return g


def _riesz(value, riesz_map=None):
    """The Function that represents a derivative in the l2, L2 (default) or H1 inner product."""
    V, g = value.space, Function(value.space)
    kind = riesz_map or "L2"
    if kind == "l2":
        _coefficients(g)[:] = value.array
        return g
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    if kind == "H1":
        a = a + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    elif kind != "L2":
        raise ValueError(f"unknown Riesz map '{riesz_map}'")
    rhs = g.gf.vec.CreateVector()
    rhs.FV().NumPy()[:] = value.array
    g.gf.vec.data = _sv.assemble(a).mat.Inverse(V.fes.FreeDofs()) * rhs
    return g


# ---- Function as a pyadjoint type

def _f_assign_numpy(dst, src, offset):
    n = len(_coefficients(dst))
    _coefficients(dst)[:] = src[offset:offset + n]
    return dst, offset + n


def _f_add(self, other):
    r = _copy(self)
    _coefficients(r)[:] += _coefficients(other)
    return r


def _f_mul(self, a):
    r = _copy(self)
    _coefficients(r)[:] *= float(a)
    return r


_FUNCTION = dict(
    _ad_create_checkpoint=_copy,
    _ad_restore_at_checkpoint=lambda self, checkpoint: checkpoint,
    _ad_init_object=classmethod(lambda cls, obj: obj),
    _ad_init_zero=lambda self, dual=False: (Dual(np.zeros(self.function_space().dim()), self.function_space())
                                            if dual else Function(self.function_space())),
    _ad_convert_riesz=lambda self, value, riesz_map=None: _riesz(value, riesz_map),
    _ad_add=_f_add,
    _ad_mul=_f_mul,
    _ad_dot=lambda self, other: float(_coefficients(self) @ _coefficients(other)),
    _ad_copy=_copy,
    _ad_dim=lambda self: self.function_space().dim(),
    _ad_assign_numpy=staticmethod(_f_assign_numpy),
    _ad_to_list=staticmethod(lambda m: _coefficients(m).tolist()),
)


# ---- Constant as a pyadjoint type

def _c_new(c, values):
    values = np.reshape(values, c.ufl_shape)
    if isinstance(c, DomainConstant):
        return DomainConstant(values, domain=c.ufl_domain().ufl_cargo())
    return GlobalConstant(values)


def _c_assign_numpy(dst, src, offset):
    n = len(dst.values())
    return _c_new(dst, src[offset:offset + n]), offset + n


def _c_values(c):
    return c.values() if isinstance(c, _CONSTANTS) else np.ravel(c)


_CONSTANT = dict(
    _ad_create_checkpoint=lambda self: _c_new(self, self.values()),
    _ad_restore_at_checkpoint=lambda self, checkpoint: checkpoint,
    _ad_init_object=classmethod(lambda cls, obj: obj if isinstance(obj, _CONSTANTS) else GlobalConstant(obj)),
    _ad_init_zero=lambda self, dual=False: _c_new(self, 0 * self.values()),
    _ad_convert_riesz=lambda self, value, riesz_map=None: _c_new(self, _c_values(value)),
    _ad_add=lambda self, other: _c_new(self, self.values() + _c_values(other)),
    _ad_mul=lambda self, a: _c_new(self, self.values() * float(a)),
    _ad_dot=lambda self, other: float(self.values() @ _c_values(other)),
    _ad_copy=lambda self: _c_new(self, self.values()),
    _ad_dim=lambda self: len(self.values()),
    _ad_assign_numpy=staticmethod(_c_assign_numpy),
    _ad_to_list=staticmethod(lambda m: _c_values(m).tolist()),
)


def _overload(cls, methods):
    if not issubclass(cls, OverloadedType):
        cls.__bases__ = cls.__bases__ + (OverloadedType,)
    cls._block_variable = Weakref()
    for name, method in methods.items():
        setattr(cls, name, method)


# ---- derivatives of forms

def _whole(f):
    if getattr(f, "_parent", None) is not None:
        raise NotImplementedError("components of a mixed or vector function (u.subfunctions, u.sub(i)) are not "
                                  "tracked on the tape; use split(u) in forms, and set them before recording")
    return f


def _dependencies(*exprs):
    """The Functions and Constants an expression or form depends on."""
    deps = []
    for e in exprs:
        if isinstance(e, (ufl.Form, ufl.core.expr.Expr)):
            deps += [_whole(c) for c in extract_coefficients(e) if isinstance(c, Function)]
            for cls in _CONSTANTS:
                deps += sorted(extract_type(e, cls), key=lambda c: c.count())
    return list(dict.fromkeys(deps))


def _derivative(functional, dep):
    """Derivative of a form without arguments with respect to a Function or Constant."""
    if isinstance(dep, Function):
        vec = _sv.assemble(ufl.derivative(functional, dep))
        return Dual(vec.get_local(), dep.function_space())
    # a constant is differentiated as a function in the space of real numbers
    R = FunctionSpace(mesh_of(functional), "R", 0, shape=dep.ufl_shape or None)
    r = Function(R)
    _coefficients(r)[:] = dep.values()
    vec = _sv.assemble(ufl.derivative(ufl.replace(functional, {dep: r}), r))
    return vec.get_local().reshape(dep.ufl_shape)


class _FormBlock(Block):
    def _mapping(self, inputs):
        return {bv.output: val for bv, val in zip(self.get_dependencies(), inputs) if bv.output is not val}


def _replace(e, mapping):
    return ufl.replace(e, mapping) if mapping and isinstance(e, (ufl.Form, ufl.core.expr.Expr)) else e


# ---- assemble of a functional

class AssembleBlock(_FormBlock):
    def __init__(self, form):
        super().__init__()
        self.form = form
        for d in _dependencies(form):
            self.add_dependency(d, no_duplicates=True)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return _sv.assemble(_replace(self.form, self._mapping(inputs)))

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        form = _replace(self.form, self._mapping(inputs))
        return _derivative(form, inputs[idx]) * float(adj_inputs[0])


def assemble(expr, *args, **kwargs):
    out = _sv.assemble(expr, *args, **kwargs)
    if annotate_tape() and isinstance(expr, ufl.Form) and not expr.arguments():
        block = AssembleBlock(expr)
        get_working_tape().add_block(block)
        out = AdjFloat(out)
        block.add_output(out.block_variable)
    return out


# ---- variational solves

class SolveBlock(_FormBlock):
    """``lhs == rhs`` solved for u; `linear` means lhs is bilinear."""

    def __init__(self, lhs, rhs, u, bcs, linear, **kwargs):
        super().__init__()
        self.lhs, self.rhs, self.u, self.bcs, self.linear, self.kwargs = lhs, rhs, u, list(bcs), linear, kwargs
        self.in_forms = _dependencies(lhs, rhs)
        for d in self.in_forms + _dependencies(*[bc.g for bc in self.bcs]):
            self.add_dependency(d, no_duplicates=True)

    def _residual(self, mapping, u):
        lhs, rhs = _replace(self.lhs, mapping), _replace(self.rhs, mapping)
        F = ufl.action(lhs, u) if self.linear else lhs
        return F if _sv._is_zero(rhs) else F - rhs

    def _solve(self, mapping, u):
        bcs = [bc.reconstruct(g=_replace(bc.g, mapping)) for bc in self.bcs]
        lhs, rhs = _replace(self.lhs, mapping), _replace(self.rhs, mapping)
        if self.linear:
            solver = _sv.LinearVariationalSolver(_sv.LinearVariationalProblem(lhs, rhs, u, bcs), **self.kwargs)
            _linear_solve(solver)
        else:
            F = lhs if _sv._is_zero(rhs) else lhs - rhs
            solver = _sv.NonlinearVariationalSolver(_sv.NonlinearVariationalProblem(F, u, bcs), **self.kwargs)
            _nonlinear_solve(solver)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        mapping = self._mapping(inputs)
        u = _copy(mapping.get(self.u, self.u)) if not self.linear else Function(self.u.function_space())
        mapping[self.u] = u
        self._solve(mapping, u)
        return u

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        mapping = self._mapping(inputs)
        u = self.get_outputs()[0].saved_output
        mapping[self.u] = u
        F = self._residual(mapping, u)
        lam = Function(u.function_space())
        if adj_inputs[0] is None:
            return F, lam
        bcs = [bc.reconstruct(g=0) for bc in self.bcs]
        A = _sv.assemble(ufl.adjoint(ufl.derivative(F, u)), bcs=bcs)
        rhs = lam.gf.vec.CreateVector()
        rhs.FV().NumPy()[:] = adj_inputs[0].array
        _sv.LinearSolver(A, self.kwargs.get("solver_parameters"), self.kwargs.get("nullspace")).solve(lam, rhs)
        return F, lam

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        F, lam = prepared
        dep = block_variable.output
        if dep is self.u:
            return None
        if dep not in self.in_forms:
            raise NotImplementedError("derivatives with respect to Dirichlet values")
        return -_derivative(ufl.action(F, lam), inputs[idx])


_linear_solve = _sv.LinearVariationalSolver.solve
_nonlinear_solve = _sv.NonlinearVariationalSolver.solve


def _annotated_solve(original, linear):
    def solve(self, bounds=None):
        if not annotate_tape():
            return original(self, bounds)
        p = self.problem
        lhs, rhs = (p.a, p.L) if linear else (p.F, None)
        block = SolveBlock(lhs, rhs, p.u, p.bcs, linear, solver_parameters=self.params, nullspace=self.nullspace)
        get_working_tape().add_block(block)
        with stop_annotating():
            original(self, bounds)
        block.add_output(p.u.create_block_variable())
    return solve


# ---- Function.assign, interpolate, copy

class AssignBlock(_FormBlock):
    """``target.assign(other)``: a copy, a number, or an expression that is affine in each Function."""

    def __init__(self, target, other):
        super().__init__()
        self.space, self.other = target.function_space(), other
        for d in _dependencies(other):
            self.add_dependency(d, no_duplicates=True)

    def _evaluate(self, mapping):
        f = Function(self.space)
        _assign(f, _replace(self.other, mapping))
        return f

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self._evaluate(self._mapping(inputs))

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        dep, adj = block_variable.output, adj_inputs[0]
        if self.other is dep:
            return adj
        if not isinstance(dep, Function) or dep.function_space() != self.space:
            raise NotImplementedError("assign: derivative with respect to a Constant or a Function of another space")
        # d(target)/d(dep) must be a multiple of the identity; find the factor by a perturbation
        mapping = self._mapping(inputs)
        r, p = Function(self.space), _copy(inputs[idx])
        _coefficients(r)[:] = np.random.default_rng(0).standard_normal(len(adj.array))
        _coefficients(p)[:] += _coefficients(r)
        w = _coefficients(self._evaluate({**mapping, dep: p})) - _coefficients(self._evaluate({**mapping, dep: inputs[idx]}))
        rv = _coefficients(r)
        factor = (w @ rv) / (rv @ rv)
        if np.linalg.norm(w - factor * rv) > 1e-8 * max(1.0, abs(factor)) * np.linalg.norm(rv):
            raise NotImplementedError("assign or interpolate of an expression that is not a linear combination "
                                      "of functions of the same space; use project")
        return adj * factor


_assign, _interpolate, _copy_method = Function.assign, Function.interpolate, Function.copy


def _annotated_assign(original):
    def assign(self, other, **kwargs):
        if not annotate_tape():
            return original(self, other, **kwargs)
        block = AssignBlock(_whole(self), other)
        get_working_tape().add_block(block)
        with stop_annotating():
            original(self, other, **kwargs)
        block.add_output(self.create_block_variable())
        return self
    return assign


def _function_copy(self, deepcopy=True):
    f = _copy_method(self, deepcopy)
    if annotate_tape():
        block = AssignBlock(f, self)
        get_working_tape().add_block(block)
        block.add_output(f.create_block_variable())
    return f


def _function_project(self, expr, **kwargs):
    return self.assign(_sv.project(expr, self.function_space(), **kwargs))


def _constant_assign(original):
    def assign(self, value):
        original(self, value)
        if annotate_tape():
            self.create_block_variable()
        return self
    return assign


# ---- switch it on

_overload(Function, _FUNCTION)
for _cls in _CONSTANTS:
    _overload(_cls, _CONSTANT)
    _cls.assign = _constant_assign(_cls.assign)

Function.assign = _annotated_assign(_assign)
Function.interpolate = _annotated_assign(_interpolate)
Function.copy = _function_copy
Function.project = _function_project
_sv.LinearVariationalSolver.solve = _annotated_solve(_linear_solve, True)
_sv.NonlinearVariationalSolver.solve = _annotated_solve(_nonlinear_solve, False)
ngsufl.assemble = assemble

if get_working_tape() is None:
    set_working_tape(Tape())

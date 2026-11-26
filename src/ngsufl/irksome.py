"""Runge-Kutta and multistep time stepping with Irksome.

    from ngsufl import *
    from ngsufl.irksome import Dt, RadauIIA, TimeStepper

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx
    stepper = TimeStepper(F, RadauIIA(2), t, dt, u, bcs=bc)
    stepper.advance()

This module is an Irksome backend, ``backend="ngsufl.irksome"``, and re-exports the
parts of Irksome that work with it.  It has to be imported before the first form is
assembled, because Irksome registers a UFL type.  Needs ``irksome`` and ``firedrake-fiat``.
"""

import importlib.util
import sys
import types
from functools import reduce
from operator import mul

import ufl

import ngsufl
from ngsufl.solving import MixedVectorSpaceBasis, VectorSpaceBasis

try:
    import petsc4py  # noqa: F401
except ModuleNotFoundError:
    # Irksome takes two constants and a print function from PETSc
    _petsc = types.ModuleType("petsc4py")
    _petsc.PETSc = types.SimpleNamespace(INFINITY=float("inf"), NINFINITY=float("-inf"),
                                         Sys=types.SimpleNamespace(Print=print))
    sys.modules["petsc4py"] = _petsc

try:
    import irksome  # noqa: F401
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        f"{err}; install with: pip install firedrake-fiat git+https://github.com/firedrakeproject/Irksome") from None

from irksome import (BDF, AdamsBashforth, AdamsMoulton, Alexander, BackwardEuler, Dt,  # noqa: E402
                     DIRKTimeStepper, GaussLegendre, LobattoIIIA, LobattoIIIC,
                     MultistepTimeStepper, PareschiRusso, QinZhang, RadauIIA,
                     StageValueTimeStepper, WSODIRK)
from irksome.stage_derivative import StageDerivativeTimeStepper  # noqa: E402
from irksome.tableaux.multistep_tableaux import MultistepTableau  # noqa: E402
from irksome.tools import get_sub  # noqa: E402

__all__ = ["Dt", "TimeStepper", "BackwardEuler", "GaussLegendre", "RadauIIA", "LobattoIIIA",
           "LobattoIIIC", "Alexander", "QinZhang", "PareschiRusso", "WSODIRK", "BDF",
           "AdamsBashforth", "AdamsMoulton", "MeshConstant", "StageDerivativeTimeStepper",
           "StageValueTimeStepper", "DIRKTimeStepper", "MultistepTimeStepper"]

_STEPPERS = {"deriv": StageDerivativeTimeStepper, "value": StageValueTimeStepper, "dirk": DIRKTimeStepper}


def TimeStepper(F, method, t, dt, u0, stage_type="deriv", **kwargs):
    """Stepper for the semi-discrete form F, which contains ``Dt(u0)``.

    `method` is a Butcher tableau (``RadauIIA(2)``, ``GaussLegendre(2)``, ``Alexander()``, ...)
    or a multistep tableau (``BDF(2)``).  `stage_type` is "deriv" (default), "value" or
    "dirk".  Keywords as in Irksome: ``bcs``, ``solver_parameters``, ``nullspace``, ...
    """
    kwargs.setdefault("backend", sys.modules[__name__])
    kwargs["bcs"] = list(extract_bcs(kwargs.get("bcs")))
    if isinstance(method, MultistepTableau):
        return MultistepTimeStepper(F, method, t, dt, u0, **kwargs)
    if stage_type not in _STEPPERS:
        raise NotImplementedError(f"stage_type '{stage_type}' with the ngsufl backend")
    return _STEPPERS[stage_type](F, method, t, dt, u0, **kwargs)


# ---- the backend interface of Irksome

assemble = ngsufl.assemble
derivative = ufl.derivative
norm = ngsufl.norm
Function = ngsufl.Function
TestFunction = ngsufl.TestFunction
TrialFunction = ngsufl.TrialFunction
DirichletBC = ngsufl.DirichletBC
Constant = ngsufl.Constant


class EquationBC:
    pass


class EquationBCSplit:
    pass


def get_stage_space(V, num_stages):
    return reduce(mul, (V for _ in range(num_stages)))


def extract_bcs(bcs):
    if bcs is None:
        return ()
    return (bcs,) if isinstance(bcs, DirichletBC) else tuple(bcs)


def get_function_space(u):
    return u.ufl_function_space()


def get_stages(V, num_stages):
    return Function(get_stage_space(V, num_stages))


class MeshConstant:
    def __init__(self, msh):
        self.msh = msh

    def Constant(self, val=0.0):
        return Constant(val)


def get_mesh_constant(MC):
    return MC.Constant if MC else Constant


def create_variational_problem(F, u, bcs=None, J=None, Jp=None, **kwargs):
    if len(F.arguments()) == 2:
        a, L = ufl.system(F)
        kwargs.pop("is_linear", None)
        return ngsufl.LinearVariationalProblem(a, L, u, bcs=bcs, **kwargs)
    return ngsufl.NonlinearVariationalProblem(F, u, bcs=bcs, J=J, **kwargs)


def create_variational_solver(problem, **kwargs):
    if isinstance(problem, ngsufl.LinearVariationalProblem):
        return ngsufl.LinearVariationalSolver(problem, **kwargs)
    return ngsufl.NonlinearVariationalSolver(problem, **kwargs)


def invalidate_jacobian(solver):
    solver.invalidate_jacobian()


def _field(bc, V):
    path = getattr(bc.space, "path", ())
    return (0, path) if len(V) == 1 else (path[0], path[1:])


def bc2space(bc, V):
    """The part of the space or function V that the boundary condition acts on."""
    return get_sub(V, getattr(bc.space, "path", ()))


def stage2spaces4bc(bc, V, Vbig, i):
    field, comp = _field(bc, V)
    return get_sub(Vbig.sub(field + len(V) * i), comp)


def create_bounds_constrained_bc(V, g, sub_domain, bounds, solver_parameters=None):
    raise NotImplementedError("bound constraints with the ngsufl backend")


def getNullspace(V, Vbig, num_stages, nullspace):
    """The null space of the stage problem: the given one, repeated for every stage."""
    if nullspace is None:
        return None
    if isinstance(nullspace, VectorSpaceBasis):
        nullspace = [(0, nullspace)]
    elif isinstance(nullspace, MixedVectorSpaceBasis):
        nullspace = [(j, b) for j, b in enumerate(nullspace) if isinstance(b, VectorSpaceBasis)]
    fields = dict(nullspace)
    if not fields or min(fields) < 0 or max(fields) >= len(V):
        raise ValueError("null space entries must be (index, VectorSpaceBasis) with a valid field index")
    if len(V) * num_stages == 1:
        return fields[0]
    bases = [fields.get(j, Vbig.sub(j + len(V) * i)) for i in range(num_stages) for j in range(len(V))]
    return MixedVectorSpaceBasis(Vbig, bases)


if importlib.util.find_spec("firedrake") is None:
    # some Irksome steppers do not pass their backend on and ask for the default one
    import irksome.backends  # noqa: E402

    sys.modules["irksome.backends.firedrake"] = irksome.backends.firedrake = sys.modules[__name__]

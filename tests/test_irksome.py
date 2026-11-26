"""Irksome time steppers on the ngsufl backend."""

import numpy as np
import pytest

pytest.importorskip("FIAT")
pytest.importorskip("ngsufl.irksome")

from ngsufl.irksome import (BDF, Alexander, BackwardEuler, Dt, GaussLegendre,  # noqa: E402
                            QinZhang, RadauIIA, TimeStepper)
from ngsufl import *  # noqa: E402,F401,F403


def heat(method, nsteps, degree=5, N=8, T=0.5, **kw):
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", degree)
    x, y = SpatialCoordinate(mesh)
    t, dt = Constant(0.0), Constant(T / nsteps)
    S = sin(pi * x) * sin(pi * y)
    uex = exp(-t) * (1 + t * t) * S
    f = (2 * t - (1 + t * t) + 2 * pi ** 2 * (1 + t * t)) * exp(-t) * S
    u, v = Function(V).interpolate(uex), TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    stepper = TimeStepper(F, method, t, dt, u, bcs=DirichletBC(V, 0, "on_boundary"), **kw)
    for _ in range(nsteps):
        stepper.advance()
        t.assign(float(t) + float(dt))
    return errornorm(uex, u)


def rate(method, steps=(2, 4, 8), **kw):
    e = np.array([heat(method, n, **kw) for n in steps])
    return np.log2(e[:-1] / e[1:])


@pytest.mark.parametrize("method,order", [(BackwardEuler(), 1), (RadauIIA(2), 3), (GaussLegendre(2), 4)],
                         ids=["BackwardEuler", "RadauIIA2", "GaussLegendre2"])
def test_stage_derivative_order(method, order):
    assert rate(method)[-1] == pytest.approx(order, abs=0.4)


def test_stage_value_matches_stage_derivative():
    a = heat(RadauIIA(2), 4)
    b = heat(RadauIIA(2), 4, stage_type="value")
    assert b == pytest.approx(a, rel=1e-6)


def test_dirk_order():
    assert rate(QinZhang(), stage_type="dirk")[-1] == pytest.approx(2, abs=0.2)
    # third order, approached slowly for a stiff problem
    assert rate(Alexander(), steps=(8, 16, 32), stage_type="dirk")[-1] > 2.6


def test_bdf2_order():
    assert rate(BDF(2), steps=(8, 16, 32))[-1] == pytest.approx(2, abs=0.3)


def test_time_dependent_boundary_data():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    t, dt = Constant(0.0), Constant(0.1)
    uex = (1 + t ** 2) * (1 + x * x + y)
    f = 2 * t * (1 + x * x + y) - 2 * (1 + t ** 2)
    u, v = Function(V).interpolate(uex), TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    stepper = TimeStepper(F, RadauIIA(2), t, dt, u, bcs=DirichletBC(V, uex, "on_boundary"))
    for _ in range(5):
        stepper.advance()
        t.assign(float(t) + float(dt))
    # quadratic in time and in space: reproduced exactly
    assert errornorm(uex, u) < 1e-10


def test_nonlinear_problem():
    mesh = UnitIntervalMesh(20)
    V = FunctionSpace(mesh, "CG", 2)
    x, = SpatialCoordinate(mesh)
    t, dt = Constant(0.0), Constant(0.05)
    uex = (1 + t) * x * (1 - x)
    f = x * (1 - x) + 2 * (1 + t) * (1 + uex ** 2) - 2 * (1 + t) ** 3 * x * (1 - x) * (1 - 2 * x) ** 2
    u, v = Function(V).interpolate(uex), TestFunction(V)
    F = inner(Dt(u), v) * dx + (1 + u ** 2) * inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    stepper = TimeStepper(F, GaussLegendre(1), t, dt, u, bcs=DirichletBC(V, 0, "on_boundary"))
    for _ in range(4):
        stepper.advance()
        t.assign(float(t) + float(dt))
    assert errornorm(uex, u) < 1e-9
    assert stepper.num_nonlinear_iterations > 0


def test_stokes_with_pressure_nullspace():
    mesh = UnitSquareMesh(6, 6)
    Z = VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    t, dt = Constant(0.0), Constant(0.1)
    uex = (1 + t * t) * as_vector((x * x * (1 - x) ** 2 * (2 * y - 6 * y * y + 4 * y ** 3),
                                   -y * y * (1 - y) ** 2 * (2 * x - 6 * x * x + 4 * x ** 3)))
    pex = (1 + t) * (x - 0.5)
    f = (2 * t / (1 + t * t)) * uex - div(grad(uex)) + grad(pex)
    z = Function(Z)
    z.subfunctions[0].interpolate(uex)
    u, p = split(z)
    v, q = TestFunctions(Z)
    F = (inner(Dt(u), v) + inner(grad(u), grad(v)) - p * div(v) - q * div(u) - inner(f, v)) * dx
    bcs = DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")
    stepper = TimeStepper(F, RadauIIA(2), t, dt, z, bcs=bcs,
                          nullspace=[(1, VectorSpaceBasis(constant=True))])
    for _ in range(3):
        stepper.advance()
        t.assign(float(t) + float(dt))
    uh, ph = z.subfunctions
    assert errornorm(uex, uh) < 2e-4
    assert norm(div(uh)) < 5e-3
    pm = assemble(ph * dx)
    assert errornorm(pex, ph - pm) < 2e-2

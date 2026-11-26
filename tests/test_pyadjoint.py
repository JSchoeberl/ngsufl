"""Derivatives through pyadjoint, checked with Taylor tests."""

import numpy as np
import pytest

pytest.importorskip("pyadjoint")

from ngsufl import *  # noqa: E402,F401,F403
from ngsufl.pyadjoint import *  # noqa: E402,F401,F403
from ngsufl.pyadjoint import Dual  # noqa: E402

pause_annotation()


@pytest.fixture(autouse=True)
def tape():
    set_working_tape(Tape())
    continue_annotation()
    yield
    pause_annotation()


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(6, 6)


def rate(Jhat, m, h):
    return taylor_test(Jhat, m, h)


def test_poisson_source_control(mesh):
    V, W = FunctionSpace(mesh, "CG", 2), FunctionSpace(mesh, "DG", 0)
    x, y = SpatialCoordinate(mesh)
    m = Function(W).interpolate(x + y)
    u, v, w = Function(V), TestFunction(V), TrialFunction(V)
    solve(inner(grad(w), grad(v)) * dx == m * v * dx, u, bcs=DirichletBC(V, 0, "on_boundary"))
    J = assemble(0.5 * (u - sin(pi * x) * sin(pi * y))**2 * dx + 1e-4 / 2 * m**2 * dx)
    assert type(J).__name__ == "AdjFloat"
    Jhat = ReducedFunctional(J, Control(m))
    assert Jhat(m) == pytest.approx(float(J), rel=1e-13)
    assert rate(Jhat, m, Function(W).interpolate(sin(3 * x) * cos(2 * y))) > 1.95


def test_nonlinear_with_function_and_constant_controls(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    m = Function(V).interpolate(1 + x * y)
    c, k = Constant(2.0), Constant((1.0, 0.5))
    u, v = Function(V), TestFunction(V)
    F = (1 + u**2) * inner(grad(u), grad(v)) * dx + c * u * v * dx - m * (k[0] + k[1] * x) * v * dx
    solve(F == 0, u, bcs=DirichletBC(V, 0, (1, 2)))
    J = assemble(u**2 * dx + c * u * ds(4) + 0.1 * m * m * dx)
    Jhat = ReducedFunctional(J, [Control(m), Control(c), Control(k)])
    h = [Function(V).interpolate(sin(2 * x) + y), Constant(0.7), Constant((0.3, -0.4))]
    assert rate(Jhat, [m, c, k], h) > 1.95


def test_gradient_against_finite_differences(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    c = Constant(1.5)
    u, v, w = Function(V), TestFunction(V), TrialFunction(V)

    def J_of(value):
        c.assign(value)
        solve(inner(grad(w), grad(v)) * dx + c * w * v * dx == (1 + c * x) * v * dx, u,
              bcs=DirichletBC(V, 0, "on_boundary"))
        return assemble(u * u * dx)

    J = J_of(1.5)
    dJ = ReducedFunctional(J, Control(c)).derivative()
    with stop_annotating():
        fd = (float(J_of(1.5 + 1e-6)) - float(J_of(1.5 - 1e-6))) / 2e-6
    assert float(dJ.values()[0]) == pytest.approx(fd, rel=1e-6)


def test_time_loop_with_reused_solver(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u0 = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    u_n = Function(V)
    u_n.assign(u0)
    w, v, uh = TrialFunction(V), TestFunction(V), Function(V)
    dt = Constant(0.05)
    problem = LinearVariationalProblem(w * v * dx + dt * inner(grad(w), grad(v)) * dx, u_n * v * dx, uh,
                                       bcs=DirichletBC(V, 0, "on_boundary"), constant_jacobian=True)
    solver = LinearVariationalSolver(problem)
    J = 0.0
    for _ in range(5):
        solver.solve()
        u_n.assign(uh)
        J += assemble(float(dt) * uh**2 * dx)
    J += assemble(inner(grad(uh), grad(uh)) * dx)
    Jhat = ReducedFunctional(J, Control(u0))
    assert Jhat(u0) == pytest.approx(float(J), rel=1e-12)
    assert rate(Jhat, u0, Function(V).interpolate(x * (1 - x) * y)) > 1.95


def test_project_linear_combination_and_inplace(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(x * x + y)
    g = project(f * f + 1, V)
    z = Function(V)
    z.assign(2 * g - 0.5 * f)
    z += f
    z2 = z.copy()
    Jhat = ReducedFunctional(assemble(z2**3 * dx), Control(f))
    assert rate(Jhat, f, Function(V).interpolate(sin(3 * x * y))) > 1.95


def test_mixed_problem(mesh):
    Z = VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(as_vector((y, x * x)))
    nu = Constant(1.0)
    z = Function(Z)
    (u, p), (v, q) = TrialFunctions(Z), TestFunctions(Z)
    a = (nu * inner(grad(u), grad(v)) - p * div(v) - q * div(u) - 1e-8 * p * q) * dx
    solve(a == inner(f, v) * dx, z, bcs=DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary"))
    uh, ph = split(z)
    Jhat = ReducedFunctional(assemble(inner(uh, uh) * dx + ph * ph * dx), [Control(f), Control(nu)])
    h = [Function(f.function_space()).interpolate(as_vector((sin(x), x * y))), Constant(0.3)]
    assert rate(Jhat, [f, nu], h) > 1.95


def test_riesz_maps(mesh):
    W = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    m = Function(W).interpolate(1 + x)
    J = assemble(m**3 * dx)
    d = ReducedFunctional(J, Control(m)).derivative()
    assert isinstance(d, Dual)
    g = ReducedFunctional(J, Control(m)).derivative(apply_riesz=True)
    # the L2 representative of d/dm int m^3 is 3 m^2, projected
    with stop_annotating():
        assert errornorm(project(3 * m * m, W), g) < 1e-10
        gl2 = ReducedFunctional(J, Control(m, riesz_map="l2")).derivative(apply_riesz=True)
        assert gl2.dat.data == pytest.approx(d.array)


def test_minimize_recovers_regularised_optimum(mesh):
    V, W = FunctionSpace(mesh, "CG", 2), FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    def state(m):
        u, w, v = Function(V), TrialFunction(V), TestFunction(V)
        solve(inner(grad(w), grad(v)) * dx == m * v * dx, u, bcs=DirichletBC(V, 0, "on_boundary"))
        return u

    with stop_annotating():
        ud = state(Function(W).interpolate(1 + sin(pi * x) * y))
    m = Function(W)
    J = assemble(0.5 * (state(m) - ud)**2 * dx + 1e-5 / 2 * m * m * dx)
    Jhat = ReducedFunctional(J, Control(m, riesz_map="l2"))
    g0 = np.linalg.norm(Jhat.derivative().array)
    m_opt = minimize(Jhat, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-11, "ftol": 1e-16})
    assert Jhat(m_opt) < 0.01 * float(J)
    assert np.linalg.norm(Jhat.derivative().array) < 1e-6 * g0


def test_unsupported_operations_raise(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(1 + x)
    g = Function(V).interpolate(f * f)
    with pytest.raises(NotImplementedError):
        ReducedFunctional(assemble(g * dx), Control(f)).derivative()
    u, v, w = Function(V), TestFunction(V), TrialFunction(V)
    solve(inner(grad(w), grad(v)) * dx == v * dx, u, bcs=DirichletBC(V, f, "on_boundary"))
    with pytest.raises(NotImplementedError):
        ReducedFunctional(assemble(u * u * dx), Control(f)).derivative()


def test_components_of_mixed_functions_raise(mesh):
    Z = FunctionSpace(mesh, "CG", 1) * FunctionSpace(mesh, "DG", 0)
    z = Function(Z)
    x, y = SpatialCoordinate(mesh)
    with pytest.raises(NotImplementedError):
        z.subfunctions[0].interpolate(x)
    with stop_annotating():
        z.subfunctions[0].interpolate(x)
    with pytest.raises(NotImplementedError):
        assemble(z.subfunctions[0] ** 2 * dx)
    u, p = split(z)
    assert float(assemble(u * u * dx)) == pytest.approx(1 / 3)


def test_no_recording_when_paused(mesh):
    pause_annotation()
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V).interpolate(SpatialCoordinate(mesh)[0])
    J = assemble(u * u * dx)
    assert type(J) is float and not get_working_tape().get_blocks()

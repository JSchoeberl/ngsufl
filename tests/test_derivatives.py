import numpy as np
import pytest

from ngsufl import *
from ngsufl.forms import residual_form
from conftest import P2D, P3D, ev, approx


def val(expr, mesh, p):
    r = ev(to_cf(expr), mesh, p)
    shp = expr.ufl_shape
    return r.reshape(shp)


def f2(x):  # scalar cubic, 2D
    return x[0] ** 2 * x[1] + 2 * x[1] ** 3 + x[0]


def g2(x):
    return (x[0] * x[1] ** 2, x[0] ** 2 + x[1])


def gf2(p):
    x, y = p
    return np.array([2 * x * y + 1, x ** 2 + 6 * y ** 2])


def hf2(p):
    x, y = p
    return np.array([[2 * y, 2 * x], [2 * x, 12 * y]])


def Jg2(p):
    x, y = p
    return np.array([[y ** 2, 2 * x * y], [2 * x, 1.0]])


@pytest.fixture(scope="module")
def sp2(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 3)
    W = VectorFunctionSpace(mesh2, "Lagrange", 3)
    return Function(V).interpolate(f2), Function(W).interpolate(g2)


def test_grad_function(mesh2, sp2):
    f, g = sp2
    for p in P2D:
        assert val(grad(f), mesh2, p) == approx(gf2(p))
        assert val(grad(g), mesh2, p) == approx(Jg2(p))
        assert val(nabla_grad(g), mesh2, p) == approx(Jg2(p).T)
        assert val(div(g), mesh2, p) == approx(np.trace(Jg2(p)))
        assert val(nabla_div(g), mesh2, p) == approx(np.trace(Jg2(p)))
        assert val(curl(g), mesh2, p) == approx(Jg2(p)[1, 0] - Jg2(p)[0, 1])
        assert val(curl(f), mesh2, p) == approx([gf2(p)[1], -gf2(p)[0]])


def test_dx_and_Dx(mesh2, sp2):
    f, g = sp2
    for p in P2D:
        for k in range(2):
            assert val(f.dx(k), mesh2, p) == approx(gf2(p)[k])
            assert val(Dx(f, k), mesh2, p) == approx(gf2(p)[k])
            assert val(g.dx(k), mesh2, p) == approx(Jg2(p)[:, k])
            assert val(g[0].dx(k), mesh2, p) == approx(Jg2(p)[0, k])
        assert val(f.dx(0, 1), mesh2, p) == approx(hf2(p)[0, 1])
        assert val(f.dx(1, 1), mesh2, p) == approx(hf2(p)[1, 1])


def test_dx_free_index(mesh2, sp2):
    f, g = sp2
    for p in P2D:
        assert val(f.dx(i) * f.dx(i), mesh2, p) == approx(gf2(p) @ gf2(p))
        assert val(g[i].dx(i), mesh2, p) == approx(np.trace(Jg2(p)))
        assert val(g[i].dx(j) * g[i].dx(j), mesh2, p) == approx(np.sum(Jg2(p) ** 2))


def test_second_derivatives(mesh2, sp2):
    f, _ = sp2
    for p in P2D:
        assert val(grad(grad(f)), mesh2, p) == approx(hf2(p))
        assert val(div(grad(f)), mesh2, p) == approx(np.trace(hf2(p)))
        assert val(nabla_div(nabla_grad(f)), mesh2, p) == approx(np.trace(hf2(p)))
        assert val(inner(grad(grad(f)), grad(grad(f))), mesh2, p) == approx(np.sum(hf2(p) ** 2))


def test_second_derivatives_vector_function(mesh2, sp2):
    _, g = sp2
    for p in P2D:
        x, y = p
        H = np.zeros((2, 2, 2))
        H[0] = [[0, 2 * y], [2 * y, 2 * x]]
        H[1] = [[2, 0], [0, 0]]
        assert val(grad(grad(g)), mesh2, p) == approx(H)
        assert val(div(grad(g)), mesh2, p) == approx(np.einsum("kii->k", H))


def test_second_derivative_p2_lagrange(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] * x[1] + x[1] ** 2)
    assert val(grad(grad(f)), mesh2, (0.3, 0.4)) == approx([[0, 1], [1, 2]])


def test_second_derivative_dg(mesh2):
    V = FunctionSpace(mesh2, "DG", 2)
    f = Function(V).interpolate(lambda x: x[0] * x[1] + x[1] ** 2)
    assert val(div(grad(f)), mesh2, (0.3, 0.4)) == approx(2.0)


# ---------- derivatives of general expressions

def test_grad_of_expressions(mesh2, sp2):
    f, g = sp2
    for p in P2D:
        x, y = p
        gf = gf2(p)
        fv = f2(p)
        assert val(grad(f * f), mesh2, p) == approx(2 * fv * gf)
        assert val(grad(sin(f)), mesh2, p) == approx(np.cos(fv) * gf)
        assert val(grad(exp(f) / (1 + f ** 2)), mesh2, p) == approx(
            (np.exp(fv) * (1 + fv ** 2) - np.exp(fv) * 2 * fv) / (1 + fv ** 2) ** 2 * gf)
        assert val(grad(f ** 3), mesh2, p) == approx(3 * fv ** 2 * gf)
        assert val(grad(sqrt(1 + f ** 2)), mesh2, p) == approx(fv / np.sqrt(1 + fv ** 2) * gf)
        assert val(grad(f * g[0]), mesh2, p) == approx(g2(p)[0] * gf + fv * Jg2(p)[0])
        assert val(div(f * g), mesh2, p) == approx(gf @ g2(p) + fv * np.trace(Jg2(p)))
        assert val(grad(ln(1 + f ** 2)), mesh2, p) == approx(2 * fv / (1 + fv ** 2) * gf)
        assert val(grad(atan(f)), mesh2, p) == approx(gf / (1 + fv ** 2))
        assert val(grad(tanh(f)), mesh2, p) == approx(gf / np.cosh(fv) ** 2)


def test_grad_of_vector_norm_squared(mesh2, sp2):
    _, g = sp2
    for p in P2D:
        assert val(grad(dot(g, g)), mesh2, p) == approx(2 * Jg2(p).T @ np.array(g2(p)))
        assert val(grad(inner(g, g)), mesh2, p) == approx(2 * Jg2(p).T @ np.array(g2(p)))


def test_grad_spatial_coordinate_expressions(mesh2):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        a, b = p
        assert val(grad(x[0] * x[1] ** 2), mesh2, p) == approx([b ** 2, 2 * a * b])
        assert val(grad(sin(x[0]) * cos(x[1])), mesh2, p) == approx(
            [np.cos(a) * np.cos(b), -np.sin(a) * np.sin(b)])
        assert val(grad(x), mesh2, p) == approx(np.eye(2))
        assert val(div(x), mesh2, p) == approx(2.0)
        assert val(div(as_vector((x[0] ** 2, x[0] * x[1]))), mesh2, p) == approx(2 * a + a)
        assert val(curl(as_vector((-x[1], x[0]))), mesh2, p) == approx(2.0)
        assert val(grad(as_vector((x[0] * x[1], x[1] ** 2))), mesh2, p) == approx([[b, a], [0, 2 * b]])
        assert val(grad(grad(x[0] ** 3 * x[1])), mesh2, p) == approx(
            [[6 * a * b, 3 * a ** 2], [3 * a ** 2, 0]])
        assert val(div(grad(exp(x[0] * x[1]))), mesh2, p) == approx(
            np.exp(a * b) * (a ** 2 + b ** 2))
        assert val(nabla_grad(as_vector((x[0] * x[1], x[1] ** 2))), mesh2, p) == approx(
            np.array([[b, a], [0, 2 * b]]).T)
        assert val(x[0].dx(0) + x[1].dx(1), mesh2, p) == approx(2.0)


def test_grad_of_constant_and_number(mesh2):
    c = Constant(mesh2, 3.0)
    assert val(grad(c), mesh2, P2D[0]) == approx([0, 0])
    cv = Constant(mesh2, [1.0, 2.0])
    assert val(div(cv), mesh2, P2D[0]) == approx(0.0)
    assert val(grad(cv), mesh2, P2D[0]) == approx(np.zeros((2, 2)))


def _alg_u(mesh):
    x = SpatialCoordinate(mesh)
    return as_vector((x[0] ** 2 * x[1], x[0] + x[1] ** 2))


ALG = [
    ("gradtr", lambda u: grad(tr(grad(u))), lambda a, b: [2 * b, 2 * a + 2], True),
    ("graddiv", lambda u: grad(div(u)), lambda a, b: [2 * b, 2 * a + 2], True),
    ("graddet", lambda u: grad(det(grad(u))), lambda a, b: [4 * b ** 2 - 2 * a, 8 * a * b], False),
    ("gradsym01", lambda u: grad(sym(grad(u))[0, 1]), lambda a, b: [a, 0], False),
    ("divgradT", lambda u: div(grad(u).T), lambda a, b: [2 * b, 2 * a + 2], False),
    ("divgrad", lambda u: div(grad(u)), lambda a, b: [2 * b, 2], False),
    ("graddot", lambda u: grad(dot(u, u)), lambda a, b: [2 * (2 * a * b * a ** 2 * b + a + b ** 2), 2 * (a ** 2 * a ** 2 * b + 2 * b * (a + b ** 2))], True),
]


@pytest.mark.parametrize("name,fu,fr,bug", ALG, ids=[a[0] for a in ALG])
def test_grad_of_tensor_algebra_coordinates(request, mesh2, name, fu, fr, bug):
    u = _alg_u(mesh2)
    for p in P2D:
        assert val(fu(u), mesh2, p) == approx(fr(*p))


def test_grad_det_exact(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((x[0] ** 2 * x[1], x[0] + x[1] ** 2))
    F = grad(u)
    for p in P2D:
        a, b = p
        d = lambda a, b: 2 * a * b * 2 * b - a ** 2 * 1  # det of [[2ab, a^2],[1, 2b]]
        eps = 1e-6
        want = [(d(a + eps, b) - d(a - eps, b)) / 2 / eps, (d(a, b + eps) - d(a, b - eps)) / 2 / eps]
        assert val(grad(det(F)), mesh2, p) == approx(want, rel=1e-5)


def test_div_of_tensor_function(mesh2):
    V = TensorFunctionSpace(mesh2, "Lagrange", 2)
    S = Function(V).interpolate(lambda x: ((x[0] * x[1], x[1] ** 2), (x[0] ** 2, x[0] * x[1])))
    p = (0.3, 0.4)
    x, y = p
    assert val(div(S), mesh2, p) == approx([y + 2 * y, 2 * x + x])


def test_grad_of_rt_function(mesh2):
    V = FunctionSpace(mesh2, "RT", 2)
    u = Function(V).interpolate(lambda x: (x[0] + 2 * x[1], 3 * x[0]))
    p = (0.3, 0.4)
    assert val(div(u), mesh2, p) == approx(1.0)
    assert val(grad(u), mesh2, p) == approx([[1, 2], [3, 0]])


def test_curl_hcurl_3d(mesh3):
    V = FunctionSpace(mesh3, "N2curl", 3)
    f = lambda x: (x[1] * x[2] ** 2, x[0] ** 2 * x[2], x[0] * x[1] ** 2)
    u = Function(V).interpolate(f)
    for p in P3D:
        x, y, z = p
        want = [2 * x * y - x ** 2, 2 * y * z - y ** 2, 2 * x * z - z ** 2]
        assert val(curl(u), mesh3, p) == approx(want)


def test_curl_of_expression_3d(mesh3):
    x = SpatialCoordinate(mesh3)
    u = as_vector((x[1] * x[2] ** 2, x[0] ** 2 * x[2], x[0] * x[1] ** 2))
    for p in P3D:
        a, b, c = p
        want = [2 * a * b - a ** 2, 2 * b * c - b ** 2, 2 * a * c - c ** 2]
        assert val(curl(u), mesh3, p) == approx(want)
        assert val(div(curl(u)), mesh3, p) == approx(0.0, abs=1e-12)
        assert val(curl(grad(x[0] * x[1] * x[2])), mesh3, p) == approx([0, 0, 0], abs=1e-12)


def test_3d_grad_scalar_vector(mesh3):
    V = FunctionSpace(mesh3, "Lagrange", 2)
    W = VectorFunctionSpace(mesh3, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] * x[1] + x[2] ** 2)
    g = Function(W).interpolate(lambda x: (x[0] * x[2], x[1] * x[1], x[0] * x[1]))
    for p in P3D:
        x, y, z = p
        assert val(grad(f), mesh3, p) == approx([y, x, 2 * z])
        assert val(grad(grad(f)), mesh3, p) == approx([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        assert val(grad(g), mesh3, p) == approx([[z, 0, x], [0, 2 * y, 0], [y, x, 0]])
        assert val(div(g), mesh3, p) == approx(z + 2 * y)
        assert val(curl(g), mesh3, p) == approx([x - 0, x - y, 0 - 0])


def test_grad_1d(mesh1):
    V = FunctionSpace(mesh1, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] ** 2)
    assert val(grad(f), mesh1, (0.3,)) == approx([0.6])
    assert val(div(grad(f)), mesh1, (0.3,)) == approx(2.0)
    assert val(f.dx(0), mesh1, (0.3,)) == approx(0.6)


def test_grad_quad(meshq):
    V = FunctionSpace(meshq, "Q", 2)
    f = Function(V).interpolate(lambda x: x[0] ** 2 * x[1])
    assert val(grad(f), meshq, (0.3, 0.4)) == approx([0.24, 0.09])
    assert val(div(grad(f)), meshq, (0.3, 0.4)) == approx(0.8)


def test_restricted_grad_interior_vs_native(mesh2):
    V = FunctionSpace(mesh2, "DG", 2)
    f = Function(V).interpolate(lambda x: x[0] ** 2 * x[1])
    cf = to_cf(grad(f)("+"))
    assert cf is not None


# ---------- variable / diff

def test_variable_diff(mesh2):
    x = SpatialCoordinate(mesh2)
    s = variable(x[0] ** 2 + x[1])
    e = diff(sin(s) * s, s)
    for p in P2D:
        v = p[0] ** 2 + p[1]
        assert val(e, mesh2, p) == approx(np.cos(v) * v + np.sin(v))


def test_variable_diff_tensor(mesh2):
    A = variable(Constant(mesh2, [[1.0, 2.0], [0.5, 3.0]]))
    psi = 0.5 * inner(A, A) + det(A)
    S = diff(psi, A)
    Av = np.array([[1.0, 2.0], [0.5, 3.0]])
    want = Av + np.linalg.det(Av) * np.linalg.inv(Av).T
    assert val(S, mesh2, P2D[0]) == approx(want)


def test_variable_diff_of_function(mesh2, sp2):
    f, _ = sp2
    s = variable(f)
    e = diff(s ** 3 + sin(s), s)
    v = f2(P2D[0])
    assert val(e, mesh2, P2D[0]) == approx(3 * v ** 2 + np.cos(v))


def test_variable_diff_vector(mesh2, sp2):
    _, g = sp2
    vg = variable(g)
    S = diff(dot(vg, vg) / 2, vg)
    assert val(S, mesh2, P2D[0]) == approx(g2(P2D[0]))


def test_variable_diff_nh_energy(mesh3):
    F = variable(Constant(mesh3, np.eye(3) + 0.1 * np.arange(9).reshape(3, 3) / 9))
    psi = 0.5 * (tr(F.T * F) - 3) - ln(det(F))
    P = diff(psi, F)
    Fv = np.eye(3) + 0.1 * np.arange(9).reshape(3, 3) / 9
    want = Fv - np.linalg.inv(Fv).T
    assert val(P, mesh3, P3D[0]) == approx(want)


# ---------- derivative() of forms

def _fd_jacobian(F, u, eps=1e-6):
    n = len(u.array)
    cols = []
    base = u.array.copy()
    for j in range(n):
        u.array[:] = base
        u.array[j] += eps
        rp = np.array(assemble(F).FV().NumPy())
        u.array[:] = base
        u.array[j] -= eps
        rm = np.array(assemble(F).FV().NumPy())
        cols.append((rp - rm) / (2 * eps))
    u.array[:] = base
    return np.array(cols).T


def _ngs_lin_jacobian(F, u):
    a = residual_form(F, u)
    a.AssembleLinearization(u.vec)
    return a.mat.ToDense().NumPy()


def _check_jacobian(F, u, du, tol=1e-6):
    J = derivative(F, u, du)
    Ja = assemble(J).ToDense().NumPy()
    Jfd = _fd_jacobian(F, u)
    Jn = _ngs_lin_jacobian(F, u)
    scale = np.abs(Ja).max()
    assert np.abs(Ja - Jfd).max() < tol * scale
    assert np.abs(Ja - Jn).max() < tol * scale
    assert scale > 0


def test_derivative_scalar_nonlinear(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 2)
    u = Function(V).interpolate(lambda x: 0.3 + x[0] * x[1] + 0.5 * x[1] ** 2)
    v, du = TestFunction(V), TrialFunction(V)
    x = SpatialCoordinate(mesh2)
    F = (1 + u ** 2) * dot(grad(u), grad(v)) * dx - sin(x[0]) * u * v * dx + exp(u) * v * ds
    _check_jacobian(F, u, du)


def test_derivative_vector_hyperelastic(mesh2):
    V = VectorFunctionSpace(mesh2, "Lagrange", 1)
    u = Function(V).interpolate(lambda x: (0.1 * x[0] * x[1], 0.05 * x[1]))
    v, du = TestFunction(V), TrialFunction(V)
    Fd = Identity(2) + grad(u)
    psi = 0.5 * (tr(Fd.T * Fd) - 2) - ln(det(Fd)) + 0.3 * ln(det(Fd)) ** 2
    F = derivative(psi * dx, u, v)
    _check_jacobian(F, u, du)


def test_derivative_mixed(mesh2):
    V = VectorFunctionSpace(mesh2, "Lagrange", 2)
    Q = FunctionSpace(mesh2, "Lagrange", 1)
    W = V * Q
    w = Function(W)
    w.sub(0).interpolate(lambda x: (0.1 * x[0] * x[1], 0.05 * x[1]))
    w.sub(1).interpolate(lambda x: 0.2 + x[0])
    u, p = split(w)
    v, q = TestFunctions(W)
    F = inner(grad(u) * u, v) * dx + (1 + p ** 2) * inner(grad(u), grad(v)) * dx - p * div(v) * dx + q * div(u) * dx
    _check_jacobian(F, w, TrialFunction(W), tol=1e-5)


def test_derivative_default_direction(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 1)
    u = Function(V).interpolate(lambda x: 1 + x[0])
    v = TestFunction(V)
    F = u ** 3 * v * dx
    J = derivative(F, u)
    Ja = assemble(J).ToDense().NumPy()
    Jn = _ngs_lin_jacobian(F, u)
    assert np.abs(Ja - Jn).max() < 1e-10


def _dg_setup(mesh2):
    V = FunctionSpace(mesh2, "DG", 1)
    u = Function(V).interpolate(lambda x: 0.5 + x[0] * x[1])
    n = FacetNormal(mesh2)
    F = (u ** 3 * TestFunction(V) * dx + jump(u) ** 2 * jump(TestFunction(V)) * dS
         + dot(avg(u) * as_vector((1.0, 0.5)), n("+")) * jump(TestFunction(V)) * dS)
    return V, u, F


def test_derivative_dg_facet_terms_ufl(mesh2):
    V, u, F = _dg_setup(mesh2)
    Ja = assemble(derivative(F, u, TrialFunction(V))).ToDense().NumPy()
    Jfd = _fd_jacobian(F, u)
    assert np.abs(Ja - Jfd).max() < 1e-6 * np.abs(Ja).max()


@pytest.mark.xfail(reason="NGSolve linearization of nonlinear skeleton terms is zero (J=None path)", strict=True)
def test_derivative_dg_facet_terms_ngs_linearization(mesh2):
    V, u, F = _dg_setup(mesh2)
    Ja = assemble(derivative(F, u, TrialFunction(V))).ToDense().NumPy()
    Jn = _ngs_lin_jacobian(F, u)
    assert np.abs(Ja - Jn).max() < 1e-6 * np.abs(Ja).max()


def test_residual_form_apply_matches_assemble_dg(mesh2):
    V, u, F = _dg_setup(mesh2)
    a = residual_form(F, u)
    r = u.vec.CreateVector()
    a.Apply(u.vec, r)
    assert np.abs(r.FV().NumPy() - assemble(F).FV().NumPy()).max() < 1e-12


def test_derivative_second_order_symmetric(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 2)
    u = Function(V).interpolate(lambda x: 0.3 + x[0] * x[1])
    v, du = TestFunction(V), TrialFunction(V)
    E = (u ** 4 / 4 + 0.5 * dot(grad(u), grad(u)) + sin(u) * u) * dx
    J = assemble(derivative(derivative(E, u, v), u, du)).ToDense().NumPy()
    assert np.abs(J - J.T).max() < 1e-12 * np.abs(J).max()
    F = derivative(E, u, v)
    Jfd = _fd_jacobian(F, u)
    assert np.abs(J - Jfd).max() < 1e-6 * np.abs(J).max()


def test_derivative_functional_gateaux(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 2)
    u = Function(V).interpolate(lambda x: 0.3 + x[0] * x[1])
    E = (u ** 3 + dot(grad(u), grad(u))) * dx
    dE = assemble(derivative(E, u, TestFunction(V)))
    d = dE.FV().NumPy()
    base = u.array.copy()
    eps = 1e-6
    for j in (0, 5, 17):
        u.array[:] = base
        u.array[j] += eps
        Ep = assemble(E)
        u.array[:] = base
        u.array[j] -= eps
        Em = assemble(E)
        assert d[j] == approx((Ep - Em) / 2 / eps, rel=1e-6)
    u.array[:] = base


def test_derivative_with_constant_coefficient_direction(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 1)
    u = Function(V).interpolate(lambda x: 1 + x[0])
    c = Constant(mesh2, 2.0)
    v = TestFunction(V)
    dd = Function(V).interpolate(lambda x: x[1])
    F = c * u ** 2 * v * dx
    J = derivative(F, u, dd)
    got = assemble(J).FV().NumPy()
    want = assemble(2 * c * u * dd * v * dx).FV().NumPy()
    assert np.abs(got - want).max() < 1e-12


def test_derivative_of_conditional(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 1)
    u = Function(V).interpolate(lambda x: x[0] - 0.5)
    v, du = TestFunction(V), TrialFunction(V)
    F = conditional(gt(u, 0), u ** 2, -u) * v * dx
    J = assemble(derivative(F, u, du)).ToDense().NumPy()
    assert J.shape[0] == V.dim()
    Jfd = _fd_jacobian(F, u, 1e-7)
    assert np.abs(J - Jfd).max() < 1e-5

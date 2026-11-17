import math

import numpy as np
import pytest
import scipy.special as sp

from ngsufl import *
from conftest import P1D, P2D, P3D, ev, approx

TOL = 1e-10


def val(expr, mesh, p):
    r = ev(to_cf(expr), mesh, p)
    shp = expr.ufl_shape
    return r.reshape(shp)


# ---------- elementary functions

ELEM = [
    ("sqrt", sqrt, np.sqrt, lambda t: 0.2 + t),
    ("exp", exp, np.exp, lambda t: t),
    ("ln", ln, np.log, lambda t: 0.2 + t),
    ("sin", sin, np.sin, lambda t: 3 * t),
    ("cos", cos, np.cos, lambda t: 3 * t),
    ("tan", tan, np.tan, lambda t: t),
    ("asin", asin, np.arcsin, lambda t: 0.9 * t),
    ("acos", acos, np.arccos, lambda t: 0.9 * t),
    ("atan", atan, np.arctan, lambda t: 4 * t - 2),
    ("sinh", sinh, np.sinh, lambda t: 2 * t),
    ("cosh", cosh, np.cosh, lambda t: 2 * t),
    ("tanh", tanh, np.tanh, lambda t: 3 * t - 1),
    ("erf", erf, sp.erf, lambda t: 2 * t - 1),
    ("abs", abs, np.abs, lambda t: 2 * t - 1),
]


@pytest.mark.parametrize("name,uf,nf,pre", ELEM, ids=[e[0] for e in ELEM])
def test_elementary(mesh2, name, uf, nf, pre):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        assert val(uf(pre(x[0]) * 1.0 + 0 * x[1]), mesh2, p) == approx(nf(pre(p[0])), abs=TOL)
        assert val(uf(pre(x[1])), mesh2, p) == approx(nf(pre(p[1])), abs=TOL)


@pytest.mark.parametrize("name,uf,nf,pre", ELEM, ids=[e[0] for e in ELEM])
def test_elementary_of_constant(mesh2, name, uf, nf, pre):
    c = Constant(mesh2, 0.37)
    assert val(uf(c), mesh2, P2D[0]) == approx(nf(0.37), abs=TOL)
    assert val(uf(Constant(mesh2, 0.37) * 1.0), mesh2, P2D[0]) == approx(nf(0.37), abs=TOL)


@pytest.mark.parametrize("name,uf,nf,pre", ELEM, ids=[e[0] for e in ELEM])
def test_elementary_of_number(mesh2, name, uf, nf, pre):
    a = pre(0.4)
    if name == "abs":
        a = -0.4
    r = uf(a)
    r = r if isinstance(r, (int, float)) else float(r)
    assert r == approx(float(nf(a)), abs=TOL)


def test_atan2(mesh2):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        assert val(atan2(x[1] - 0.5, x[0] - 0.5), mesh2, p) == approx(math.atan2(p[1] - 0.5, p[0] - 0.5), abs=TOL)
        assert val(atan2(x[0] - 0.5, -x[1]), mesh2, p) == approx(math.atan2(p[0] - 0.5, -p[1]), abs=TOL)


def test_atan2_numbers():
    assert float(atan2(1.0, -1.0)) == approx(math.atan2(1, -1))


@pytest.mark.parametrize("pw", [2, 3, 0.5, -1, 1.7, 4])
def test_power_number(mesh2, pw):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        assert val((x[0] + 0.5) ** pw, mesh2, p) == approx((p[0] + 0.5) ** pw, rel=1e-12)


def test_power_expression(mesh2):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        assert val((x[0] + 0.5) ** (x[1] + 0.5), mesh2, p) == approx((p[0] + 0.5) ** (p[1] + 0.5), rel=1e-12)
        assert val(2.0 ** x[0], mesh2, p) == approx(2.0 ** p[0], rel=1e-12)
        assert val(x[0] ** x[0], mesh2, p) == approx(p[0] ** p[0], rel=1e-12)


def test_arith(mesh2):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        a, b = p
        assert val(x[0] + x[1] * 3 - 2, mesh2, p) == approx(a + 3 * b - 2)
        assert val(x[0] / (1 + x[1]), mesh2, p) == approx(a / (1 + b))
        assert val(1 / (x[0] * x[1]), mesh2, p) == approx(1 / (a * b))
        assert val(-x[0] * x[1] / 3.0, mesh2, p) == approx(-a * b / 3)
        assert val((x[0] + 1) * (x[1] + 2) * (x[0] - 3), mesh2, p) == approx((a + 1) * (b + 2) * (a - 3))


# ---------- conditionals

COND = [
    ("lt", lambda a, b: lt(a, b), lambda a, b: a < b),
    ("le", lambda a, b: le(a, b), lambda a, b: a <= b),
    ("gt", lambda a, b: gt(a, b), lambda a, b: a > b),
    ("ge", lambda a, b: ge(a, b), lambda a, b: a >= b),
    ("eq", lambda a, b: eq(a, b), lambda a, b: a == b),
    ("ne", lambda a, b: ne(a, b), lambda a, b: a != b),
]


@pytest.mark.parametrize("name,uc,nc", COND, ids=[c[0] for c in COND])
@pytest.mark.parametrize("thr", [0.5, 0.3, 0.7])
def test_conditional_cmp(mesh2, name, uc, nc, thr):
    x = SpatialCoordinate(mesh2)
    e = conditional(uc(x[0], thr), 1.0 + x[1], -2.0)
    for p in [(0.3, 0.4), (0.7, 0.2), (0.5, 0.5), (0.9, 0.1)]:
        want = (1.0 + p[1]) if nc(p[0], thr) else -2.0
        assert val(e, mesh2, p) == approx(want), (p, thr)


def test_conditional_equal_points(mesh2):
    x = SpatialCoordinate(mesh2)
    p = (0.5, 0.5)
    for u, want in [(lt(x[0], 0.5), 0), (le(x[0], 0.5), 1), (gt(x[0], 0.5), 0), (ge(x[0], 0.5), 1),
                    (eq(x[0], 0.5), 1), (ne(x[0], 0.5), 0)]:
        assert val(conditional(u, 1.0, 0.0), mesh2, p) == want


def test_conditional_logic(mesh2):
    x = SpatialCoordinate(mesh2)
    c1, c2 = lt(x[0], 0.5), gt(x[1], 0.3)
    exprs = {
        "and": (And(c1, c2), lambda a, b: a < .5 and b > .3),
        "or": (Or(c1, c2), lambda a, b: a < .5 or b > .3),
        "not": (Not(c1), lambda a, b: not a < .5),
        "nested": (And(Or(c1, Not(c2)), Not(And(c1, c2))), lambda a, b: (a < .5 or not b > .3) and not (a < .5 and b > .3)),
    }
    pts = [(0.3, 0.4), (0.7, 0.2), (0.3, 0.2), (0.7, 0.9)]
    for name, (c, f) in exprs.items():
        e = conditional(c, 1.0, 0.0)
        for p in pts:
            assert val(e, mesh2, p) == float(f(*p)), (name, p)


def test_conditional_tensor_valued(mesh2):
    x = SpatialCoordinate(mesh2)
    e = conditional(lt(x[0], 0.5), as_vector((x[0], 1.0)), as_vector((0.0, x[1])))
    assert val(e, mesh2, (0.3, 0.4)) == approx([0.3, 1.0])
    assert val(e, mesh2, (0.7, 0.4)) == approx([0.0, 0.4])


def test_conditional_nested_in_math(mesh2):
    x = SpatialCoordinate(mesh2)
    e = sin(conditional(gt(x[0], x[1]), x[0], x[1]))
    for p in P2D:
        assert val(e, mesh2, p) == approx(math.sin(max(p)))


def test_minmax(mesh2):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        assert val(min_value(x[0], x[1]), mesh2, p) == approx(min(p))
        assert val(max_value(x[0], x[1]), mesh2, p) == approx(max(p))
        assert val(max_value(x[0], 0.5), mesh2, p) == approx(max(p[0], 0.5))
        assert val(min_value(0.5, x[1]), mesh2, p) == approx(min(p[1], 0.5))


def test_minmax_numbers():
    assert float(max_value(1.0, 2.0)) == 2.0
    assert float(min_value(1.0, 2.0)) == 1.0


# ---------- tensor algebra

A2 = np.array([[1.0, 2.0], [3.0, 5.0]])
B2 = np.array([[0.5, -1.0], [2.0, 0.25]])
A3 = np.array([[1.0, 2.0, 0.5], [3.0, 5.0, -1.0], [0.2, 0.7, 4.0]])
B3 = np.array([[0.5, -1.0, 2.0], [2.0, 0.25, 1.0], [1.5, 0.3, -2.0]])
a3 = np.array([1.0, -2.0, 0.5])
b3 = np.array([0.3, 0.8, -1.2])


def Cm(mesh, M):
    return Constant(mesh, M)


def test_inner_dot_ranks(mesh2):
    m = mesh2
    s, v, w = Cm(m, 2.5), Cm(m, [1.0, -2.0]), Cm(m, [0.5, 3.0])
    A, B = Cm(m, A2), Cm(m, B2)
    sv, vv, wv = 2.5, np.array([1.0, -2.0]), np.array([0.5, 3.0])
    p = P2D[0]
    assert val(inner(s, s), m, p) == approx(sv * sv)
    assert val(inner(v, w), m, p) == approx(vv @ wv)
    assert val(inner(A, B), m, p) == approx(np.sum(A2 * B2))
    assert val(dot(s, s), m, p) == approx(sv * sv)
    assert val(dot(v, w), m, p) == approx(vv @ wv)
    assert val(dot(A, v), m, p) == approx(A2 @ vv)
    assert val(dot(v, A), m, p) == approx(vv @ A2)
    assert val(dot(A, B), m, p) == approx(A2 @ B2)


def test_rank3_dot_inner(mesh2):
    m = mesh2
    T = np.arange(8.0).reshape(2, 2, 2) + 1
    Tc = as_tensor(T.tolist())
    v = Cm(m, [1.0, -2.0])
    A = Cm(m, A2)
    p = P2D[0]
    assert val(dot(Tc, v), m, p) == approx(np.einsum("ijk,k->ij", T, [1.0, -2.0]))
    assert val(dot(v, Tc), m, p) == approx(np.einsum("k,kij->ij", [1.0, -2.0], T))
    assert val(dot(Tc, A), m, p) == approx(np.einsum("ijk,kl->ijl", T, A2))
    assert val(inner(Tc, Tc), m, p) == approx(np.sum(T * T))
    assert val(outer(v, A), m, p) == approx(np.einsum("i,jk->ijk", [1.0, -2.0], A2))
    assert val(outer(A, v), m, p) == approx(np.einsum("ij,k->ijk", A2, [1.0, -2.0]))


def test_rank3_index_notation(mesh2):
    m = mesh2
    T = np.arange(8.0).reshape(2, 2, 2) + 1
    Tc = as_tensor(T.tolist())
    v = Cm(m, [1.0, -2.0])
    p = P2D[0]
    e = as_tensor(Tc[i, j, k] * v[k], (i, j))
    assert val(e, m, p) == approx(np.einsum("ijk,k->ij", T, [1.0, -2.0]))
    e = as_vector(Tc[i, j, j], i)
    assert val(e, m, p) == approx(np.einsum("ijj->i", T))
    e = as_tensor(Tc[k, j, i], (i, j, k))
    assert val(e, m, p) == approx(T.transpose(2, 1, 0))
    assert val(Tc[1, :, 0], m, p) == approx(T[1, :, 0])


@pytest.mark.parametrize("dim", [2, 3])
def test_outer_cross_tr_det(mesh2, mesh3, dim):
    m = mesh2 if dim == 2 else mesh3
    p = P2D[0] if dim == 2 else P3D[0]
    A, B = (A2, B2) if dim == 2 else (A3, B3)
    a, b = (np.array([1.0, -2.0]), np.array([0.3, 0.8])) if dim == 2 else (a3, b3)
    Ac, Bc, ac, bc = Cm(m, A), Cm(m, B), Cm(m, a), Cm(m, b)
    assert val(outer(ac, bc), m, p) == approx(np.outer(a, b))
    assert val(tr(Ac), m, p) == approx(np.trace(A))
    assert val(det(Ac), m, p) == approx(np.linalg.det(A))
    assert val(inv(Ac), m, p) == approx(np.linalg.inv(A))
    assert val(dev(Ac), m, p) == approx(A - np.trace(A) / dim * np.eye(dim))
    assert val(sym(Ac), m, p) == approx((A + A.T) / 2)
    assert val(skew(Ac), m, p) == approx((A - A.T) / 2)
    assert val(transpose(Ac), m, p) == approx(A.T)
    assert val(Ac.T, m, p) == approx(A.T)
    assert val(cofac(Ac), m, p) == approx(np.linalg.det(A) * np.linalg.inv(A).T)
    assert val(Identity(dim), m, p) == approx(np.eye(dim))
    assert val(Ac * ac, m, p) == approx(A @ a)
    assert val(Ac * Bc, m, p) == approx(A @ B)
    assert val(Ac + 2 * Bc - Ac / 2, m, p) == approx(A / 2 + 2 * B)


def test_cross_3d(mesh3):
    ac, bc = Cm(mesh3, a3), Cm(mesh3, b3)
    assert val(cross(ac, bc), mesh3, P3D[0]) == approx(np.cross(a3, b3))


def test_cross_2d_raises(mesh2):
    # UFL gives cross() the shape (3,) whatever the operands are
    with pytest.raises(ValueError):
        to_cf(cross(Cm(mesh2, [1.0, -2.0]), Cm(mesh2, [0.3, 0.8])))


def test_perp(mesh2):
    a = np.array([1.0, -2.0])
    assert val(perp(Cm(mesh2, a)), mesh2, P2D[0]) == approx([2.0, 1.0])


def test_dev_det_inv_scalar_mixed_with_x(mesh2):
    x = SpatialCoordinate(mesh2)
    M = as_matrix([[x[0], 1.0], [x[1], 2 + x[0]]])
    for p in P2D:
        Mn = np.array([[p[0], 1.0], [p[1], 2 + p[0]]])
        assert val(det(M), mesh2, p) == approx(np.linalg.det(Mn))
        assert val(inv(M), mesh2, p) == approx(np.linalg.inv(Mn))
        assert val(tr(M * M), mesh2, p) == approx(np.trace(Mn @ Mn))
        assert val(M.T * M, mesh2, p) == approx(Mn.T @ Mn)
        assert val(inner(M, M.T), mesh2, p) == approx(np.sum(Mn * Mn.T))


def test_det_inv_1x1_and_scalar(mesh2):
    x = SpatialCoordinate(mesh2)
    assert val(det(as_matrix([[x[0] + 1]])), mesh2, P2D[0]) == approx(1.3)
    assert val(inv(as_matrix([[x[0] + 1]])), mesh2, P2D[0]) == approx([[1 / 1.3]])


def test_nested_tensor_ops(mesh3):
    A, B = Cm(mesh3, A3), Cm(mesh3, B3)
    p = P3D[0]
    e = inner(dev(sym(A * B)), skew(B.T * A) + sym(A))
    S = (A3 @ B3 + (A3 @ B3).T) / 2
    D = S - np.trace(S) / 3 * np.eye(3)
    K = B3.T @ A3
    want = np.sum(D * ((K - K.T) / 2 + (A3 + A3.T) / 2))
    assert val(e, mesh3, p) == approx(want)
    assert val(det(inv(A) * B), mesh3, p) == approx(np.linalg.det(np.linalg.inv(A3) @ B3))


# ---------- index notation

def test_index_basic(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((x[0], x[1] ** 2))
    v = as_vector((1 + x[1], 2.0))
    for p in P2D:
        un, vn = np.array([p[0], p[1] ** 2]), np.array([1 + p[1], 2.0])
        assert val(u[i] * v[i], mesh2, p) == approx(un @ vn)
        assert val(u[i] * u[i], mesh2, p) == approx(un @ un)
        assert val(as_tensor(u[i] * v[j], (i, j))[1, 0], mesh2, p) == approx(un[1] * vn[0])
        assert val(as_tensor(u[i] * v[j], (i, j)), mesh2, p) == approx(np.outer(un, vn))
        assert val(as_tensor(u[i] * v[j], (j, i)), mesh2, p) == approx(np.outer(vn, un))


@pytest.mark.parametrize("dim", [2, 3])
def test_index_matrix(mesh2, mesh3, dim):
    m = mesh2 if dim == 2 else mesh3
    p = P2D[0] if dim == 2 else P3D[0]
    A, B, a = (A2, B2, np.array([1.0, -2.0])) if dim == 2 else (A3, B3, a3)
    Ac, Bc, ac = Cm(m, A), Cm(m, B), Cm(m, a)
    assert val(as_tensor(Ac[i, j] * Bc[j, k], (i, k)), m, p) == approx(A @ B)
    assert val(as_tensor(Ac[i, j] * Bc[k, j], (i, k)), m, p) == approx(A @ B.T)
    assert val(as_tensor(Ac[j, i] * Bc[j, k], (i, k)), m, p) == approx(A.T @ B)
    assert val(as_tensor(Ac[j, i] * Bc[k, j], (i, k)), m, p) == approx(A.T @ B.T)
    assert val(as_tensor(Ac[i, j] * Bc[j, k], (k, i)), m, p) == approx((A @ B).T)
    assert val(Ac[i, i], m, p) == approx(np.trace(A))
    assert val(Ac[i, j] * Bc[i, j], m, p) == approx(np.sum(A * B))
    assert val(Ac[i, j] * Bc[j, i], m, p) == approx(np.sum(A * B.T))
    assert val(as_vector(Ac[i, j] * ac[j], i), m, p) == approx(A @ a)
    assert val(as_vector(Ac[j, i] * ac[j], i), m, p) == approx(A.T @ a)
    assert val(ac[i] * Ac[i, j] * ac[j], m, p) == approx(a @ A @ a)
    assert val(as_tensor(Ac[i, j], (j, i)), m, p) == approx(A.T)
    assert val(as_vector(Ac[i, i] * ac[j], j), m, p) == approx(np.trace(A) * a)
    assert val(as_tensor(Ac[i, j] * Bc[j, k] * Ac[k, l], (i, l)), m, p) == approx(A @ B @ A)
    assert val(Ac[i, j] * Bc[j, k] * Ac[k, i], m, p) == approx(np.trace(A @ B @ A))


def test_index_slices_and_fixed(mesh3):
    m, p = mesh3, P3D[0]
    A = Cm(m, A3)
    assert val(A[0, :], m, p) == approx(A3[0])
    assert val(A[:, 1], m, p) == approx(A3[:, 1])
    assert val(A[2, 1], m, p) == approx(A3[2, 1])
    assert val(A[:, 0] + A[0, :], m, p) == approx(A3[:, 0] + A3[0])
    assert val(A[0, i] * A[i, 1], m, p) == approx(A3[0] @ A3[:, 1])
    assert val(A[i, 0] * A[i, 2], m, p) == approx(A3[:, 0] @ A3[:, 2])
    assert val(as_vector(A[i, 1], i), m, p) == approx(A3[:, 1])


def test_as_vector_matrix_constructors(mesh2):
    x = SpatialCoordinate(mesh2)
    for p in P2D:
        assert val(as_vector((x[0], x[1], x[0] * x[1])), mesh2, p) == approx([p[0], p[1], p[0] * p[1]])
        M = as_matrix([[x[0], 0], [1, x[1]]])
        assert val(M, mesh2, p) == approx(np.array([[p[0], 0], [1, p[1]]]))
        assert val(as_matrix([[1, 2, 3], [4, 5, 6]]), mesh2, p) == approx(np.array([[1, 2, 3], [4, 5, 6]]))
        assert val(as_vector([x[0], 0]), mesh2, p) == approx([p[0], 0])
        assert val(as_vector([0, 0]), mesh2, p) == approx([0, 0])


def test_nested_index_sums(mesh3):
    m, p = mesh3, P3D[1]
    A, B = Cm(m, A3), Cm(m, B3)
    e = A[i, j] * B[j, k] * A[k, i] + A[i, i] * B[j, j]
    want = np.trace(A3 @ B3 @ A3) + np.trace(A3) * np.trace(B3)
    assert val(e, m, p) == approx(want)
    e = (A[i, j] + B[i, j]) * (A[i, j] - B[i, j])
    assert val(e, m, p) == approx(np.sum(A3 ** 2) - np.sum(B3 ** 2))
    e = A[i, j] * (B[j, k] + A[j, k]) * B[k, i]
    assert val(e, m, p) == approx(np.trace(A3 @ (B3 + A3) @ B3))
    e = sin(A[i, i]) * B[0, 0]
    assert val(e, m, p) == approx(math.sin(np.trace(A3)) * B3[0, 0])
    e = sqrt(A[i, j] * A[i, j])
    assert val(e, m, p) == approx(np.linalg.norm(A3))


def test_index_with_functions_of_x(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((sin(x[0]), cos(x[1])))
    M = as_matrix([[x[0], x[1]], [x[1] ** 2, 1 + x[0]]])
    for p in P2D:
        un = np.array([math.sin(p[0]), math.cos(p[1])])
        Mn = np.array([[p[0], p[1]], [p[1] ** 2, 1 + p[0]]])
        assert val(as_vector(M[i, j] * u[j], i), mesh2, p) == approx(Mn @ un)
        assert val(u[i] * M[i, j] * u[j] / (1 + x[0]), mesh2, p) == approx(un @ Mn @ un / (1 + p[0]))
        assert val(exp(u[i] * u[i]), mesh2, p) == approx(math.exp(un @ un))
        assert val(as_vector(u[i] / (1 + M[i, i]), i), mesh2, p) == approx(un / (1 + np.trace(Mn)))


def test_scalar_times_tensor(mesh2):
    x = SpatialCoordinate(mesh2)
    A = Cm(mesh2, A2)
    p = P2D[0]
    assert val(x[0] * A, mesh2, p) == approx(p[0] * A2)
    assert val(A * x[0], mesh2, p) == approx(p[0] * A2)
    assert val(A / (1 + x[1]), mesh2, p) == approx(A2 / (1 + p[1]))
    assert val(sin(x[0]) * as_vector((1, x[1])), mesh2, p) == approx(math.sin(p[0]) * np.array([1, p[1]]))
    assert val(-A, mesh2, p) == approx(-A2)
    assert val(2 * A - A, mesh2, p) == approx(A2)


def test_power_and_functions_of_indexed_tensors(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((x[0] + 1, x[1] + 2))
    p = P2D[0]
    un = np.array([p[0] + 1, p[1] + 2])
    assert val(as_vector(exp(u[0]) * u[i], i), mesh2, p) == approx(math.exp(un[0]) * un)


def test_abs_of_tensor_component_expr(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((x[0] - 0.5, x[1] - 0.5))
    assert val(as_vector(abs(u[i]), i), mesh2, P2D[0]) == approx([0.2, 0.1])


def test_conditional_with_indexed(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((x[0] - 0.5, x[1] - 0.5))
    e = conditional(gt(u[0], 0), u[0], -u[1])
    assert val(e, mesh2, (0.7, 0.2)) == approx(0.2)
    assert val(e, mesh2, (0.3, 0.2)) == approx(0.3)


def test_conditional_on_index_sum(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((x[0], x[1]))
    e = conditional(lt(u[i] * u[i], 0.25), 1.0, 0.0)
    assert val(e, mesh2, (0.3, 0.2)) == 1.0
    assert val(e, mesh2, (0.6, 0.6)) == 0.0


# ---------- Functions and Constants

def test_function_values(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: 1 + x[0] * x[1] + x[0] ** 2)
    for p in P2D:
        assert val(f, mesh2, p) == approx(1 + p[0] * p[1] + p[0] ** 2)
        assert val(sin(f) + f ** 2, mesh2, p) == approx(math.sin(1 + p[0] * p[1] + p[0] ** 2) + (1 + p[0] * p[1] + p[0] ** 2) ** 2)


def test_vector_function_values(mesh2):
    V = VectorFunctionSpace(mesh2, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: (x[0] * x[1], 1 + x[0] ** 2))
    for p in P2D:
        fn = np.array([p[0] * p[1], 1 + p[0] ** 2])
        assert val(f, mesh2, p) == approx(fn)
        assert val(f[0] * f[1], mesh2, p) == approx(fn[0] * fn[1])
        assert val(dot(f, f), mesh2, p) == approx(fn @ fn)
        assert val(outer(f, f), mesh2, p) == approx(np.outer(fn, fn))
        assert val(f[i] * f[i], mesh2, p) == approx(fn @ fn)


def test_tensor_function_values(mesh2):
    V = TensorFunctionSpace(mesh2, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: ((x[0], x[1]), (1 + x[0], 2 * x[1])))
    for p in P2D:
        Fn = np.array([[p[0], p[1]], [1 + p[0], 2 * p[1]]])
        assert val(f, mesh2, p) == approx(Fn)
        assert val(det(f), mesh2, p) == approx(np.linalg.det(Fn))
        assert val(f.T * f, mesh2, p) == approx(Fn.T @ Fn)


def test_mixed_function_split(mesh2):
    V = VectorFunctionSpace(mesh2, "Lagrange", 2)
    Q = FunctionSpace(mesh2, "Lagrange", 1)
    W = V * Q
    w = Function(W)
    w.sub(0).interpolate(lambda x: (x[0], x[1]))
    w.sub(1).interpolate(lambda x: 1 + x[0])
    u, q = split(w)
    for p in P2D:
        assert val(u, mesh2, p) == approx(p)
        assert val(q, mesh2, p) == approx(1 + p[0])
        assert val(u[0] * q, mesh2, p) == approx(p[0] * (1 + p[0]))
        assert val(w[2], mesh2, p) == approx(1 + p[0])
        assert val(w, mesh2, p) == approx([p[0], p[1], 1 + p[0]])


def test_constant_changes(mesh2):
    c = Constant(mesh2, 2.0)
    cf = to_cf(c * 3)
    assert ev(cf, mesh2, P2D[0]) == approx(6.0)
    c.assign(5.0)
    assert ev(cf, mesh2, P2D[0]) == approx(15.0)


def test_spatial_coordinate_3d(mesh3):
    x = SpatialCoordinate(mesh3)
    p = P3D[0]
    assert val(x, mesh3, p) == approx(p)
    assert val(dot(x, x), mesh3, p) == approx(np.dot(p, p))
    assert val(cross(x, as_vector((0, 0, 1))), mesh3, p) == approx(np.cross(p, [0, 0, 1]))


def test_1d_expressions(mesh1):
    x = SpatialCoordinate(mesh1)
    for p in P1D:
        assert val(sin(x[0]) * x[0], mesh1, p) == approx(math.sin(p[0]) * p[0])
        assert val(x, mesh1, p) == approx(p)


def test_quad_mesh_expression(meshq):
    x = SpatialCoordinate(meshq)
    assert val(x[0] * x[1], meshq, P2D[0]) == approx(0.12)


def test_cf_errors_for_free_args(mesh2):
    V = FunctionSpace(mesh2, "Lagrange", 1)
    with pytest.raises(Exception):
        to_cf(TestFunction(V))


def test_unsupported_node_message(mesh2):
    x = SpatialCoordinate(mesh2)
    with pytest.raises(NotImplementedError):
        to_cf(bessel_J(0, x[0]))


def test_index_sum_of_constant_zero_dimension(mesh2):
    x = SpatialCoordinate(mesh2)
    e = as_vector((x[0] * 0, 1.0))[i] * as_vector((0, 1.0))[i]
    assert val(e, mesh2, P2D[0]) == approx(1.0)


def test_sum_of_lazy_with_different_order(mesh2):
    A, B = Cm(mesh2, A2), Cm(mesh2, B2)
    e = as_tensor(A[i, j] + B[j, i], (i, j))
    assert val(e, mesh2, P2D[0]) == approx(A2 + B2.T)


def test_division_of_lazy(mesh2):
    A = Cm(mesh2, A2)
    x = SpatialCoordinate(mesh2)
    e = as_tensor(A[i, j] / (1 + x[0]), (i, j))
    assert val(e, mesh2, P2D[0]) == approx(A2 / 1.3)


def test_inner_with_index_inside(mesh2):
    A, B = Cm(mesh2, A2), Cm(mesh2, B2)
    e = inner(as_tensor(A[i, j] * B[j, k], (i, k)), A)
    assert val(e, mesh2, P2D[0]) == approx(np.sum((A2 @ B2) * A2))


def test_complex_free_names_ok(mesh2):
    e = as_tensor(Cm(mesh2, A2)[l, k], (k, l))
    assert val(e, mesh2, P2D[0]) == approx(A2.T)


def test_abs_of_vector_expression(mesh2):
    x = SpatialCoordinate(mesh2)
    u = as_vector((x[0] - 0.5, x[1] - 0.5))
    assert val(abs(u), mesh2, P2D[0]) == approx([0.2, 0.1])


def test_abs_of_matrix_constant(mesh2):
    assert val(abs(Cm(mesh2, -A2)), mesh2, P2D[0]) == approx(np.abs(A2))


@pytest.mark.parametrize("which", ["rowslice", "colslice", "reversed", "nested"])
def test_index_sum_of_sliced_tensor(mesh2, which):
    A, w = Cm(mesh2, A2), Cm(mesh2, [0.5, -1.5])
    wn = np.array([0.5, -1.5])
    p = P2D[0]
    if which == "rowslice":
        assert val(A[k, :] * w[k], mesh2, p) == approx(wn @ A2)
    elif which == "colslice":
        assert val(A[:, k] * w[k], mesh2, p) == approx(A2 @ wn)
    elif which == "reversed":
        assert val(w[k] * A[k, :], mesh2, p) == approx(wn @ A2)
    else:
        assert val(dot(A, A[k, :] * w[k]), mesh2, p) == approx(A2 @ (wn @ A2))


def test_index_sum_of_sliced_tensor_component_works(mesh2):
    A, w = Cm(mesh2, A2), Cm(mesh2, [0.5, -1.5])
    assert val((A[k, :] * w[k])[1], mesh2, P2D[0]) == approx((np.array([0.5, -1.5]) @ A2)[1])


def test_listtensor_with_index_sums(mesh2):
    A, w = Cm(mesh2, A2), Cm(mesh2, [0.5, -1.5])
    wn = np.array([0.5, -1.5])
    assert val(as_vector([A[0, k] * w[k], A[1, k] * w[k]]), mesh2, P2D[0]) == approx(A2 @ wn)
    want = np.array([[A2[0] @ wn, 1], [2, A2[1] @ wn]])
    assert val(as_matrix([[A[0, k] * w[k], 1.0], [2.0, A[1, k] * w[k]]]), mesh2, P2D[0]) == approx(want)


def test_compound_ops_of_index_built_matrix(mesh2):
    A = Cm(mesh2, A2)
    M = as_matrix(A[i, k] * A[j, k], (i, j))
    Mn = A2 @ A2.T
    p = P2D[0]
    assert val(det(M), mesh2, p) == approx(np.linalg.det(Mn))
    assert val(inv(M), mesh2, p) == approx(np.linalg.inv(Mn))
    assert val(dev(M), mesh2, p) == approx(Mn - np.trace(Mn) / 2 * np.eye(2))
    assert val(M.T, mesh2, p) == approx(Mn.T)
    assert val(tr(M), mesh2, p) == approx(np.trace(Mn))
    assert val(inner(M, A), mesh2, p) == approx(np.sum(Mn * A2))
    assert val(sqrt(inner(A, A)), mesh2, p) == approx(np.linalg.norm(A2))


def test_ufl_elementwise_and_misc_ops(mesh2):
    x = SpatialCoordinate(mesh2)
    p = P2D[0]
    assert val(sign(x[0] - 0.5), mesh2, p) == approx(-1.0)
    assert val(sign(x[0] - 0.1), mesh2, p) == approx(1.0)
    u, w = as_vector((x[0], 2.0)), as_vector((3.0, x[1]))
    assert val(elem_mult(u, w), mesh2, p) == approx([0.9, 0.8])
    assert val(elem_div(u, w), mesh2, p) == approx([0.1, 5.0])
    assert val(elem_pow(u, w), mesh2, p) == approx([0.027, 2 ** 0.4])
    assert val(elem_op(sin, u), mesh2, p) == approx([math.sin(0.3), math.sin(2.0)])
    M = as_matrix(((1.0, 2.0), (3.0, x[0])))
    assert val(diag(M), mesh2, p) == approx([[1, 0], [0, 0.3]])
    assert val(diag_vector(M), mesh2, p) == approx([1, 0.3])


def test_permutation_symbol_and_cross_index(mesh3):
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.2, -1.0, 4.0])
    ac, bc = Cm(mesh3, a), Cm(mesh3, b)
    assert val(as_vector(PermutationSymbol(3)[i, j, k] * ac[j] * bc[k], i), mesh3, P3D[0]) == approx(np.cross(a, b))
    assert val(as_vector(cross(ac, bc)[i] * 2.0, i), mesh3, P3D[0]) == approx(2 * np.cross(a, b))


def test_variable_derivative_of_translated_expression(mesh2):
    x = SpatialCoordinate(mesh2)
    s = variable(x[0])
    e = diff(conditional(gt(s, 0.1), s ** 2, 0.0), s)
    assert val(e, mesh2, (0.3, 0.4)) == approx(0.6)
    assert val(e, mesh2, (0.05, 0.4)) == approx(0.0)


def test_grad_of_nonsmooth_and_special_functions(mesh2):
    x = SpatialCoordinate(mesh2)
    p = (0.3, 0.4)
    a, b = p
    assert val(grad(conditional(lt(x[0], 0.5), x[0] ** 2, x[1])), mesh2, p) == approx([0.6, 0.0])
    assert val(grad(abs(x[0] - 0.5)), mesh2, p) == approx([-1.0, 0.0])
    assert val(grad(max_value(x[0], x[1])), mesh2, p) == approx([0.0, 1.0])
    assert val(grad(x[0] ** x[1]), mesh2, p) == approx([b * a ** (b - 1), a ** b * math.log(a)])
    assert val(grad(atan2(x[1], x[0])), mesh2, p) == approx([-b / (a * a + b * b), a / (a * a + b * b)])
    assert val(grad(erf(x[0])), mesh2, p) == approx([2 / math.sqrt(math.pi) * math.exp(-a * a), 0.0])
    assert val(grad(sqrt(x[0])), mesh2, p) == approx([0.5 / math.sqrt(a), 0.0])
    e = tan(x[0]) + asin(x[0]) + acos(x[0]) + sinh(x[0]) + cosh(x[0]) + tanh(x[0]) + ln(x[0]) + atan(x[0])
    want = 1 / math.cos(a) ** 2 + math.cosh(a) + math.sinh(a) + 1 / math.cosh(a) ** 2 + 1 / a + 1 / (1 + a * a)
    assert val(grad(e), mesh2, p) == approx([want, 0.0])

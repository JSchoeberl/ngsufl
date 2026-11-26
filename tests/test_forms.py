import math

import numpy as np
import ngsolve as ngs
import pytest

from ngsufl import *
from conftest import approx, close, dense, native_bf, native_lf


def A(form, **kw):
    return dense(assemble(form, **kw))


def vec(form):
    return assemble(form).FV().NumPy().copy()


# ---------- volume forms on all cell types

MESHES = {
    "tri": lambda: UnitSquareMesh(3, 3),
    "quad": lambda: UnitSquareMesh(3, 3, quadrilateral=True),
    "tet": lambda: BoxMesh(2, 2, 2),
    "hex": lambda: BoxMesh(2, 2, 2, hexahedral=True),
    "seg": lambda: UnitIntervalMesh(5),
}
_cache = {}


def mk(name):
    if name not in _cache:
        _cache[name] = MESHES[name]()
    return _cache[name]


@pytest.mark.parametrize("cell", list(MESHES))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_mass_stiffness(cell, k):
    m = mk(cell)
    V = FunctionSpace(m, "Lagrange", k)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(u * v * dx), native_bf(V.fes, lambda U, W: U * W * ngs.dx))
    close(A(inner(grad(u), grad(v)) * dx), native_bf(V.fes, lambda U, W: ngs.grad(U) * ngs.grad(W) * ngs.dx))
    close(A(u.dx(0) * v * dx), native_bf(V.fes, lambda U, W: ngs.grad(U)[0] * W * ngs.dx))


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex"])
def test_elasticity(cell):
    m = mk(cell)
    d = m.dim
    V = VectorFunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    mu, lam = 1.3, 0.7
    eps = lambda w: sym(grad(w))
    sig = lambda w: lam * tr(eps(w)) * Identity(d) + 2 * mu * eps(w)
    a = inner(sig(u), eps(v)) * dx
    E = lambda w: ngs.Sym(ngs.Grad(w))
    S = lambda w: lam * ngs.Trace(E(w)) * ngs.Id(d) + 2 * mu * E(w)
    close(A(a), native_bf(V.fes, lambda U, W: ngs.InnerProduct(S(U), E(W)) * ngs.dx))


@pytest.mark.parametrize("cell", ["tri", "tet", "quad"])
def test_convection(cell):
    m = mk(cell)
    d = m.dim
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    b = as_vector([1.0 + x[0], 2.0 - x[1]] + [0.5 * x[0]] * (d - 2))
    bn = ngs.CF(tuple([1.0 + ngs.x, 2.0 - ngs.y] + [0.5 * ngs.x] * (d - 2)))
    close(A(dot(b, grad(u)) * v * dx), native_bf(V.fes, lambda U, W: (bn * ngs.grad(U)) * W * ngs.dx))
    close(A(inner(b * u, grad(v)) * dx), native_bf(V.fes, lambda U, W: (bn * U) * ngs.grad(W) * ngs.dx))
    bc = Constant(m, [1.0, 2.0] + [0.5] * (d - 2))
    close(A(dot(bc, grad(u)) * v * dx), native_bf(V.fes, lambda U, W: (ngs.CF(tuple([1.0, 2.0] + [0.5] * (d - 2))) * ngs.grad(U)) * W * ngs.dx))


def test_variable_coefficient_diffusion():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    kappa = 1 + x[0] ** 2 + x[1]
    close(A(kappa * inner(grad(u), grad(v)) * dx),
          native_bf(V.fes, lambda U, W: (1 + ngs.x ** 2 + ngs.y) * ngs.grad(U) * ngs.grad(W) * ngs.dx(bonus_intorder=10)), 1e-10)


def test_function_coefficient_in_form():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    k = Function(V).interpolate(lambda x: 1 + x[0] * x[1])
    u, v = TrialFunction(V), TestFunction(V)
    close(A(k * k * u * v * dx), native_bf(V.fes, lambda U, W: k.gf * k.gf * U * W * ngs.dx(bonus_intorder=10)), 1e-10)
    k.assign(2.0)
    close(A(k * u * v * dx), native_bf(V.fes, lambda U, W: 2.0 * U * W * ngs.dx))


def test_constant_in_form_tracks_value():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    c = Constant(m, 2.0)
    u, v = TrialFunction(V), TestFunction(V)
    M0 = A(u * v * dx)
    close(A(c * u * v * dx), 2 * M0)
    c.assign(5.0)
    close(A(c * u * v * dx), 5 * M0)


@pytest.mark.parametrize("cell", ["tri", "tet", "quad", "hex", "seg"])
def test_linear_forms(cell):
    m = mk(cell)
    V = FunctionSpace(m, "Lagrange", 2)
    v = TestFunction(V)
    x = SpatialCoordinate(m)
    f = 1 + x[0] * x[0] if m.dim == 1 else 1 + x[0] * x[1]
    fn = (1 + ngs.x * ngs.x) if m.dim == 1 else 1 + ngs.x * ngs.y
    close(vec(f * v * dx), native_lf(V.fes, lambda W: fn * W * ngs.dx), 1e-10)
    close(vec(inner(Constant(m, [1.0, 2, 3][:m.dim]), grad(v)) * dx),
          native_lf(V.fes, lambda W: ngs.CF(tuple([1.0, 2, 3][:m.dim])) * ngs.grad(W) * ngs.dx))


def test_vector_linear_form():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    v = TestFunction(V)
    f = Constant(m, (1.0, -2.0))
    close(vec(dot(f, v) * dx), native_lf(V.fes, lambda W: ngs.CF((1.0, -2.0)) * W * ngs.dx))


def test_sum_of_integrals_and_scalings():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    M, K = A(u * v * dx), A(inner(grad(u), grad(v)) * dx)
    close(A(2 * u * v * dx - 3 * inner(grad(u), grad(v)) * dx + u * v * dx), 3 * M - 3 * K)
    close(A((u * v + inner(grad(u), grad(v))) * dx), M + K)


# ---------- mixed forms

def test_stokes_blocks():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    a = inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx
    nat = native_bf(W.fes, lambda U, T: ngs.InnerProduct(ngs.Grad(U[0]), ngs.Grad(T[0])) * ngs.dx
                    - ngs.div(T[0]) * U[1] * ngs.dx - T[1] * ngs.div(U[0]) * ngs.dx)
    close(A(a), nat)


def test_stokes_stabilised_blocks():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 1)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    h = CellDiameter(m)
    a = inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx + h ** 2 * inner(grad(p), grad(q)) * dx
    hc = mk("tri").cell_quantity("CellDiameter")
    nat = native_bf(W.fes, lambda U, T: ngs.InnerProduct(ngs.Grad(U[0]), ngs.Grad(T[0])) * ngs.dx
                    - ngs.div(T[0]) * U[1] * ngs.dx - T[1] * ngs.div(U[0]) * ngs.dx
                    + hc * hc * ngs.grad(U[1]) * ngs.grad(T[1]) * ngs.dx)
    close(A(a), nat)


def test_rt_dg_mixed():
    m = mk("tri")
    W = FunctionSpace(m, "RT", 2) * FunctionSpace(m, "DG", 1)
    (s, u), (t, v) = TrialFunctions(W), TestFunctions(W)
    a = (dot(s, t) + div(t) * u + div(s) * v) * dx
    nat = native_bf(W.fes, lambda U, T: (U[0] * T[0] + ngs.div(T[0]) * U[1] + ngs.div(U[0]) * T[1]) * ngs.dx)
    close(A(a), nat)


def test_mixed_linear_form_and_split_components():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    v, q = TestFunctions(W)
    L = dot(Constant(m, (1.0, 2.0)), v) * dx + 3 * q * dx
    nat = native_lf(W.fes, lambda T: ngs.CF((1.0, 2.0)) * T[0] * ngs.dx + 3 * T[1] * ngs.dx)
    close(vec(L), nat)
    w = TestFunction(W)
    L2 = (w[0] + 2 * w[1] + 3 * w[2]) * dx
    nat = native_lf(W.fes, lambda T: (T[0][0] + 2 * T[0][1] + 3 * T[1]) * ngs.dx)
    close(vec(L2), nat)


def test_rectangular_form_and_adjoint():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    u, q = TrialFunction(V), TestFunction(Q)
    B = A(div(u) * q * dx)
    assert B.shape == (Q.dim(), V.dim())
    nat = native_bf(Q.fes, lambda U, W: ngs.div(U) * W * ngs.dx, trial=V.fes)
    close(B, nat)
    a = div(u) * q * dx
    close(A(adjoint(a)), B.T)


def test_mixed_space_dg_and_h1():
    m = mk("tri")
    W = FunctionSpace(m, "Lagrange", 2) * FunctionSpace(m, "DG", 1)
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    a = u * v * dx + p * q * dx + u * q * dx
    nat = native_bf(W.fes, lambda U, T: (U[0] * T[0] + U[1] * T[1] + U[0] * T[1]) * ngs.dx)
    close(A(a), nat)


# ---------- UFL form algebra

def test_lhs_rhs_system():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    f = Constant(m, 3.0)
    F = u * v * dx + inner(grad(u), grad(v)) * dx - f * v * dx
    M, K = A(u * v * dx), A(inner(grad(u), grad(v)) * dx)
    close(A(lhs(F)), M + K)
    close(vec(rhs(F)), vec(f * v * dx))
    a, L = system(F)
    close(A(a), M + K)
    close(vec(L), vec(f * v * dx))


def test_action():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    w = Function(V).interpolate(lambda x: x[0] ** 2 + x[1])
    a = (inner(grad(u), grad(v)) + 2 * u * v) * dx
    want = w.vec.CreateVector()
    want.data = assemble(a) * w.vec
    close(vec(action(a, w)), want.FV().NumPy())


def test_action_of_linear_form_is_functional():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    v = TestFunction(V)
    w = Function(V).interpolate(lambda x: x[0] ** 2 + x[1])
    x = SpatialCoordinate(m)
    val = assemble(action(x[0] * v * dx, w))
    assert val == approx(0.5)


def test_adjoint_nonsymmetric():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    a = dot(Constant(m, (1.0, 0.5)), grad(u)) * v * dx
    Ma = A(a)
    assert np.abs(Ma - Ma.T).max() > 1e-3
    close(A(adjoint(a)), Ma.T)


def test_replace():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    v = TestFunction(V)
    w1 = Function(V).interpolate(lambda x: x[0])
    w2 = Function(V).interpolate(lambda x: x[1] ** 2)
    F = w1 ** 2 * v * dx
    close(vec(replace(F, {w1: w2})), vec(w2 ** 2 * v * dx))
    u = TrialFunction(V)
    a = u * v * dx
    close(vec(replace(a, {u: w2})), vec(w2 * v * dx))


def test_derivative_of_form_arguments_numbering():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    u = Function(V).interpolate(lambda x: 1 + x[0])
    v, du = TestFunction(V), TrialFunction(V)
    J = derivative(u ** 2 * v * dx, u, du)
    close(A(J), A(2 * u * du * v * dx))


# ---------- boundary terms

def nds(m, name='.*'):
    return ngs.dx(element_boundary=True, definedon=m.ngs.Boundaries(name))


@pytest.mark.parametrize("sel,name", [("bottom", "bottom"), (3, "bottom"), (4, "top"), ("left|right", "left|right"),
                                       ((3, 4), "bottom|top"), ((2, 1), "right|left"), ("top|bottom", "top|bottom")])
def test_ds_selectors(sel, name):
    m = mk("tri")
    assert m.boundaries[:4] == ("bottom", "right", "top", "left")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    close(A(u * v * ds(sel)), native_bf(V.fes, lambda U, W: U * W * ngs.ds(name)))
    close(vec((1 + x[0]) * v * ds(sel)), native_lf(V.fes, lambda W: (1 + ngs.x) * W * ngs.ds(name)))


def test_ds_all_and_mixed_ids():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(u * v * ds), native_bf(V.fes, lambda U, W: U * W * ngs.ds))
    a = u * v * ds(1) + 2 * u * v * ds("right") + 3 * u * v * ds((3, 4))
    nat = native_bf(V.fes, lambda U, W: U * W * ngs.ds("left") + 2 * U * W * ngs.ds("right") + 3 * U * W * ngs.ds("bottom|top"))
    close(A(a), nat)


def _poly(d):
    if d == 1:
        return (lambda x: x[0] ** 2 + x[0]), ngs.CF((2 * ngs.x + 1,)), 2.0
    if d == 2:
        return (lambda x: x[0] ** 2 + x[0] * x[1] + 2 * x[1]), ngs.CF((2 * ngs.x + ngs.y, ngs.x + 2)), 2.0
    return (lambda x: x[0] ** 2 + x[0] * x[1] + 2 * x[1] + x[2] ** 2 + x[1] * x[2]), \
        ngs.CF((2 * ngs.x + ngs.y, ngs.x + 2 + ngs.z, 2 * ngs.z + ngs.y)), 4.0


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex", "seg"])
def test_ds_normal_derivative_terms(cell):
    """Integration by parts for a smooth f: int_bnd (df/dn) g = int (lap f) g + grad f . grad g."""
    m = mk(cell)
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(m)
    pf, gradf, lapf = _poly(m.dim)
    f = Function(V).interpolate(pf)
    rng = np.random.default_rng(1)
    g = rng.standard_normal(V.dim())
    gg = ngs.GridFunction(V.fes)
    gg.vec.FV().NumPy()[:] = g
    rhs = ngs.Integrate(((lapf * gg + gradf * ngs.grad(gg))), m.ngs, order=6)
    assert g @ A(dot(grad(u), n) * v * ds) @ f.array == approx(rhs, rel=1e-10)


def pf_cf(d):
    if d == 1:
        return ngs.x ** 2 + ngs.x
    if d == 2:
        return ngs.x ** 2 + ngs.x * ngs.y + 2 * ngs.y
    return ngs.x ** 2 + ngs.x * ngs.y + 2 * ngs.y + ngs.z ** 2 + ngs.y * ngs.z


def test_ds_normal_derivative_selected_boundary():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(m)
    parts = sum(A(dot(grad(u), n) * v * ds(i)) for i in (1, 2, 3, 4))
    close(parts, A(dot(grad(u), n) * v * ds), 1e-12)


@pytest.mark.parametrize("cell", ["tet", "hex"])
def test_ds_3d_names(cell):
    m = mk(cell)
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(u * v * ds("front")), native_bf(V.fes, lambda U, W: U * W * ngs.ds("front")))
    close(A(u * v * ds("top|bottom")), native_bf(V.fes, lambda U, W: U * W * ngs.ds("top|bottom")))


def test_ds_1d():
    m = mk("seg")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(u * v * ds("left")), native_bf(V.fes, lambda U, W: U * W * ngs.ds("left")))
    close(A(u * v * ds(2)), native_bf(V.fes, lambda U, W: U * W * ngs.ds("right")))


def test_ds_out_of_range_id():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    with pytest.raises(Exception):
        assemble(TrialFunction(V) * TestFunction(V) * ds(9))


def test_dx_subdomain_by_name_and_id():
    from netgen.occ import OCCGeometry, WorkPlane
    wp = WorkPlane().Rectangle(2, 1).Face()
    wp.faces.name = "A"
    left = WorkPlane().Rectangle(1, 1).Face()
    left.faces.name = "A"
    right = WorkPlane().MoveTo(1, 0).Rectangle(1, 1).Face()
    right.faces.name = "B"
    from netgen.occ import Glue
    g = Glue([left, right])
    m = Mesh(OCCGeometry(g, dim=2).GenerateMesh(maxh=0.5))
    assert set(m.materials) == {"A", "B"}
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    ia, ib = m.materials.index("A") + 1, m.materials.index("B") + 1
    close(A(u * v * dx("A")), native_bf(V.fes, lambda U, W: U * W * ngs.dx("A")))
    close(A(u * v * dx(ib)), native_bf(V.fes, lambda U, W: U * W * ngs.dx("B")))
    close(A(u * v * dx((ia, ib))), native_bf(V.fes, lambda U, W: U * W * ngs.dx))
    assert assemble(Constant(m, 1.0) * dx("A")) == approx(1.0)
    assert assemble(Constant(m, 1.0) * dx("B")) == approx(1.0)
    assert assemble(Constant(m, 1.0) * dx) == approx(2.0)


# ---------- DG / interior facet forms

def dgfes(V):
    return V.ngs_space(True)


def dgnat(V, fn, **kw):
    fes = dgfes(V)
    U, W = fes.TnT()
    a = ngs.BilinearForm(fes, **kw)
    a += fn(U, W)
    a.Assemble()
    return dense(a.mat)


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex"])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_dg_jump_jump(cell, k):
    m = mk(cell)
    V = FunctionSpace(m, "DG", k)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(jump(u) * jump(v) * dS), dgnat(V, lambda U, W: (U - U.Other()) * (W - W.Other()) * ngs.dx(skeleton=True)))


@pytest.mark.parametrize("cell", ["tri", "quad", "tet"])
def test_dg_sipg_blocks(cell):
    m = mk(cell)
    d = m.dim
    V = FunctionSpace(m, "DG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(m)
    h = CellDiameter(m)
    alpha = 7.0
    a = (-inner(avg(grad(u)), jump(v, n)) * dS - inner(jump(u, n), avg(grad(v))) * dS
         + alpha / avg(h) * inner(jump(u, n), jump(v, n)) * dS)
    hc = m.cell_quantity("CellDiameter")
    nn = ngs.specialcf.normal(d)

    def nat(U, W):
        jn = lambda w: w - w.Other()
        gavg = lambda w: 0.5 * (ngs.grad(w) + ngs.grad(w).Other())
        jvn = (W - W.Other()) * nn
        jun = (U - U.Other()) * nn
        return (-(gavg(U) * jvn) - (jun * gavg(W)) + alpha / (0.5 * (hc + hc.Other())) * (jun * jvn)) * ngs.dx(skeleton=True)
    close(A(a), dgnat(V, nat), 1e-10)


def test_dg_restrictions_plus_minus():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(u("+") * v("+") * dS), dgnat(V, lambda U, W: U * W * ngs.dx(skeleton=True)))
    close(A(u("-") * v("-") * dS), dgnat(V, lambda U, W: U.Other() * W.Other() * ngs.dx(skeleton=True)))
    close(A(u("+") * v("-") * dS), dgnat(V, lambda U, W: U * W.Other() * ngs.dx(skeleton=True)))
    close(A(u("-") * v("+") * dS), dgnat(V, lambda U, W: U.Other() * W * ngs.dx(skeleton=True)))
    close(A(avg(u) * jump(v) * dS), dgnat(V, lambda U, W: 0.5 * (U + U.Other()) * (W - W.Other()) * ngs.dx(skeleton=True)))
    close(A(avg(u) * avg(v) * dS), dgnat(V, lambda U, W: 0.25 * (U + U.Other()) * (W + W.Other()) * ngs.dx(skeleton=True)))


def test_dg_normal_orientation():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(m)
    nn = ngs.specialcf.normal(2)
    b = Constant(m, (1.0, 0.3))
    bn = ngs.CF((1.0, 0.3))
    close(A(dot(b, n("+")) * u("+") * jump(v) * dS),
          dgnat(V, lambda U, W: (bn * nn) * U * (W - W.Other()) * ngs.dx(skeleton=True)))
    close(A(dot(b, n("-")) * u("-") * jump(v) * dS),
          dgnat(V, lambda U, W: (bn * (-nn)) * U.Other() * (W - W.Other()) * ngs.dx(skeleton=True)))
    close(A(jump(u, n)[0] * jump(v) * dS),
          dgnat(V, lambda U, W: ((U - U.Other()) * nn[0]) * (W - W.Other()) * ngs.dx(skeleton=True)))
    close(A(dot(jump(u, n), jump(v, n)) * dS),
          dgnat(V, lambda U, W: ((U - U.Other()) * (W - W.Other())) * (nn * nn) * ngs.dx(skeleton=True)))


def test_dg_upwind_advection():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(m)
    b = Constant(m, (1.0, 0.3))
    bn = ngs.CF((1.0, 0.3))
    nn = ngs.specialcf.normal(2)
    bnor = dot(b, n)
    un = (bnor + abs(bnor)) / 2
    a = -u * dot(b, grad(v)) * dx + dot(jump(v), un("+") * u("+") - un("-") * u("-")) * dS
    bnn = bn * nn
    upw = lambda t: (t + ngs.IfPos(t, t, -t)) / 2

    def nat(U, W):
        return (-U * (bn * ngs.grad(W))) * ngs.dx + (W - W.Other()) * (upw(bnn) * U - upw(-bnn) * U.Other()) * ngs.dx(skeleton=True)
    close(A(a), dgnat(V, nat), 1e-10)


def test_dg_cell_diameter_values():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 0)
    v = TestFunction(V)
    h = CellDiameter(m)
    got = vec(h * v * dx)
    areas = np.array(ngs.Integrate(ngs.CF(1.0), m.ngs, element_wise=True))
    pts = np.array([list(p.point) for p in m.ngs.vertices])
    hs = np.zeros(m.ngs.ne)
    for el in m.ngs.Elements(ngs.VOL):
        P = pts[[v_.nr for v_ in el.vertices]]
        hs[el.nr] = max(np.linalg.norm(a - b) for a in P for b in P)
    assert np.sort(got) == approx(np.sort(hs * areas))


def test_cell_geometric_quantities():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "DG", 0)
    v = TestFunction(V)
    a = 0.5 * 0.5 / 2
    assert np.allclose(vec(CellVolume(m) * v * dx), a * a)
    assert np.allclose(vec(CellDiameter(m) * v * dx), math.sqrt(0.5) * a)
    assert np.allclose(vec(MaxCellEdgeLength(m) * v * dx), math.sqrt(0.5) * a)
    assert np.allclose(vec(MinCellEdgeLength(m) * v * dx), 0.5 * a)
    assert np.allclose(vec(Circumradius(m) * v * dx), math.sqrt(0.5) / 2 * a)


def test_cell_quantities_3d():
    m = BoxMesh(2, 2, 2)
    V = FunctionSpace(m, "DG", 0)
    v = TestFunction(V)
    vol = vec(v * dx)
    ratio = lambda q: vec(q * v * dx) / vol
    assert np.allclose(ratio(Circumradius(m)), math.sqrt(3) / 4)
    assert ratio(CellDiameter(m)).max() == approx(math.sqrt(3) / 2)
    assert np.allclose(ratio(CellVolume(m)), vol)
    assert ratio(MaxCellEdgeLength(m)).max() == approx(math.sqrt(3) / 2)
    assert ratio(MinCellEdgeLength(m)).min() == approx(0.5)


def test_cell_diameter_hex_and_quad():
    mh = BoxMesh(2, 2, 2, hexahedral=True)
    v = TestFunction(FunctionSpace(mh, "DG", 0))
    assert np.allclose(vec(CellDiameter(mh) * v * dx), math.sqrt(3) / 2 / 8)
    mq = UnitSquareMesh(3, 3, quadrilateral=True)
    v = TestFunction(FunctionSpace(mq, "DG", 0))
    assert np.allclose(vec(CellDiameter(mq) * v * dx), math.sqrt(2) / 3 / 9)
    assert np.allclose(vec(CellVolume(mq) * v * dx), 1 / 81)


def test_linear_form_with_dS():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1)
    v = TestFunction(V)
    x = SpatialCoordinate(m)
    f = 1 + x[0] * x[1]
    fes = dgfes(V)
    fn = 1 + ngs.x * ngs.y
    lf = ngs.LinearForm(fes)
    lf += fn * (fes.TestFunction() - fes.TestFunction().Other()) * ngs.dx(skeleton=True)
    lf.Assemble()
    close(vec(f * jump(v) * dS), lf.vec.FV().NumPy(), 1e-10)
    lf2 = ngs.LinearForm(fes)
    lf2 += fn * 0.5 * (fes.TestFunction() + fes.TestFunction().Other()) * ngs.dx(skeleton=True)
    lf2.Assemble()
    close(vec(f * avg(v) * dS), lf2.vec.FV().NumPy(), 1e-10)


def test_linear_form_dS_plus_volume():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1)
    v = TestFunction(V)
    L = v * dx + jump(v) * dS + v * ds
    close(vec(L), vec(v * dx) + vec(jump(v) * dS) + vec(v * ds))


def test_dS_sums_with_volume_and_boundary_bilinear():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = u * v * dx + jump(u) * jump(v) * dS + u * v * ds
    close(A(a), A(u * v * dx) + A(jump(u) * jump(v) * dS) + A(u * v * ds))
    nat = dgnat(V, lambda U, W: U * W * ngs.dx + (U - U.Other()) * (W - W.Other()) * ngs.dx(skeleton=True))
    close(A(u * v * dx + jump(u) * jump(v) * dS), nat)


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex"])
@pytest.mark.parametrize("k", [1, 2])
def test_boundary_form_on_dg_space_quadratic_form(cell, k):
    m = mk(cell)
    V = FunctionSpace(m, "DG", k)
    g = lambda x: 1 + x[0] + 2 * x[1] + (x[0] * x[1] if k >= 2 else 0 * x[0])
    w = Function(V).interpolate(g)
    u, v = TrialFunction(V), TestFunction(V)
    want = assemble(w * w * ds)
    Mb = A(u * v * ds)
    assert w.array @ Mb @ w.array == approx(want, rel=1e-10)
    assert vec(v * ds) @ w.array == approx(assemble(w * ds), rel=1e-10)
    xs = SpatialCoordinate(m)
    gx = 1 + xs[0] + 2 * xs[1] + (xs[0] * xs[1] if k >= 2 else 0)
    exact = assemble(gx ** 2 * ds)
    assert want == approx(exact, rel=1e-10)


def test_dS_on_continuous_space_zero_jump():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u = Function(V).interpolate(lambda x: x[0] ** 2 + x[1])
    Q = FunctionSpace(m, "DG", 0)
    q = TestFunction(Q)
    assert np.abs(vec(jump(u) * jump(q) * dS)).max() < 1e-12
    assert np.abs(vec(jump(grad(u), FacetNormal(m)) * jump(q) * dS)).max() < 1e-12


def test_dS_dof_layout_matches_user_space():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    Mdg = A(jump(u) * jump(v) * dS)
    assert Mdg.shape == (V.dim(), V.dim())


def test_dS_with_user_dgjumps_space():
    m = mk("tri")
    V = FunctionSpace(m, "DG", 1, dgjumps=True)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(jump(u) * jump(v) * dS), native_bf(V.fes, lambda U, W: (U - U.Other()) * (W - W.Other()) * ngs.dx(skeleton=True)))


def test_dS_on_dg_vector_space():
    m = mk("tri")
    V = VectorFunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(m)
    nn = ngs.specialcf.normal(2)
    a = dot(jump(u), jump(v)) * dS + dot(avg(u), n("+")) * dot(avg(v), n("+")) * dS
    fes = V.ngs_space(True)
    nat = native_bf(fes, lambda U, W: ((U - U.Other()) * (W - W.Other())
                                       + (0.5 * (U + U.Other()) * nn) * (0.5 * (W + W.Other()) * nn)) * ngs.dx(skeleton=True))
    close(A(a), nat)


def test_functional_dS():
    m = UnitSquareMesh(3, 3)
    inner_len = 4.0 + 3 * math.sqrt(2) * 3 / 3 * 1.0
    assert assemble(Constant(m, 1.0) * dS(domain=m)) == approx(4.0 + 3 * math.sqrt(2))
    assert assemble(Constant(m, 1.0) * dS(domain=m)) == approx(inner_len)


def test_functional_dS_jump_of_dg_function():
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(lambda x: x[0] * 0).interpolate(conditional(lt(SpatialCoordinate(m)[0], 1 / 3), 0.0, conditional(lt(SpatialCoordinate(m)[0], 2 / 3), 1.0, 2.0)))
    assert assemble(abs(jump(f)) * dS) == approx(2.0)


def test_functional_dS_continuous_function_has_no_jump():
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] ** 2 + x[1])
    assert assemble(jump(f) ** 2 * dS) == approx(0.0, abs=1e-20)


def test_functional_dS_with_normals():
    m = UnitSquareMesh(3, 3)
    x = SpatialCoordinate(m)
    n = FacetNormal(m)
    assert assemble(dot(n("+"), n("-")) * dS) == approx(-(4.0 + 3 * math.sqrt(2)))
    V = FunctionSpace(m, "DG", 1)
    f = Function(V).interpolate(lambda x: 2 * x[0] + x[1])
    assert assemble(jump(f, n)[0] ** 2 * dS) == approx(0.0, abs=1e-20)
    assert assemble(avg(f) * dS) == approx(assemble((2 * x[0] + x[1]) * dS(domain=m)), rel=1e-12)


# ---------- functionals

@pytest.mark.parametrize("cell,vol", [("tri", 1.0), ("quad", 1.0), ("tet", 1.0), ("hex", 1.0), ("seg", 1.0)])
def test_functional_volume(cell, vol):
    m = mk(cell)
    assert assemble(Constant(m, 1.0) * dx) == approx(vol)
    x = SpatialCoordinate(m)
    prod = x[0]
    for k in range(1, m.dim):
        prod = prod * x[k]
    assert assemble(prod * dx) == approx(0.5 ** m.dim)
    assert assemble(sin(x[0]) * dx(degree=8)) == approx(1 - math.cos(1.0), rel=1e-8)


def test_functional_dx_quadrature_degree():
    m = mk("tri")
    x = SpatialCoordinate(m)
    f = x[0] ** 4
    ex = 0.2
    assert assemble(f * dx) == approx(ex, rel=1e-12)
    assert assemble(f * dx(degree=4)) == approx(ex, rel=1e-12)
    assert assemble(f * dx(metadata={"quadrature_degree": 4})) == approx(ex, rel=1e-12)
    low2 = assemble(f * dx(degree=2))
    assert abs(low2 - ex) > 1e-5
    assert low2 == approx(ngs.Integrate(ngs.x ** 4, m.ngs, order=2), rel=1e-10)


@pytest.mark.parametrize("deg", [0, 1])
def test_functional_dx_low_degree_respected(deg):
    m = mk("tri")
    x = SpatialCoordinate(m)
    got = assemble(x[0] ** 4 * dx(degree=deg))
    assert got == approx(ngs.Integrate(ngs.x ** 4, m.ngs, order=deg), rel=1e-10)


def test_dx_degree_in_forms():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    lo = vec(x[0] ** 3 * v * dx(degree=0))
    hi = vec(x[0] ** 3 * v * dx(degree=4))
    nat_hi = native_lf(V.fes, lambda W: ngs.x ** 3 * W * ngs.dx(bonus_intorder=3))
    close(hi, nat_hi, 1e-12)
    assert np.abs(lo - hi).max() > 1e-4
    Ml = A(u * v * dx(degree=1))
    assert np.abs(Ml - A(u * v * dx)).max() > 1e-4


def test_ds_quadrature_degree_in_functional():
    m = mk("tri")
    x = SpatialCoordinate(m)
    assert assemble(x[0] ** 4 * ds("bottom")) == approx(0.2, rel=1e-12)
    assert assemble(x[0] ** 4 * ds("bottom", degree=4)) == approx(0.2, rel=1e-12)
    assert abs(assemble(x[0] ** 4 * ds("bottom", degree=1)) - 0.2) > 1e-5


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex"])
def test_functional_boundary_measures(cell):
    m = mk(cell)
    d = m.dim
    one = Constant(m, 1.0)
    assert assemble(one * ds) == approx(2.0 * d if d == 2 else 6.0)
    names = m.boundaries
    for nm in names:
        assert assemble(one * ds(nm)) == approx(1.0)
    for i in range(1, len(names) + 1):
        assert assemble(one * ds(i)) == approx(1.0)
    assert assemble(one * ds((1, 2))) == approx(2.0)
    assert assemble(one * ds("|".join(names[:3]))) == approx(3.0)


def test_functional_normal_derivative():
    for cell in ("tri", "quad", "tet", "hex"):
        m = mk(cell)
        d = m.dim
        x = SpatialCoordinate(m)
        n = FacetNormal(m)
        u = x[0] ** 2 + x[1] + (x[2] ** 2 if d == 3 else 0)
        divtheorem = assemble(div(grad(u)) * dx(domain=m))
        assert assemble(dot(grad(u), n) * ds) == approx(divtheorem, abs=1e-11)
        assert divtheorem == approx(2.0 + (2.0 if d == 3 else 0.0))


def test_functional_normal_derivative_per_face():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u = Function(V).interpolate(lambda x: x[0] ** 2 + x[1])
    n = FacetNormal(m)
    assert assemble(dot(grad(u), n) * ds("right")) == approx(2.0)
    assert assemble(dot(grad(u), n) * ds("left")) == approx(0.0, abs=1e-12)
    assert assemble(dot(grad(u), n) * ds("top")) == approx(1.0)
    assert assemble(dot(grad(u), n) * ds("bottom")) == approx(-1.0)
    assert assemble(u * ds("top")) == approx(1 / 3 + 1)


def test_functional_flux_of_vector_function():
    m = mk("tet")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    w = Function(V).interpolate(lambda x: (x[0] ** 2, x[1], 2 * x[2]))
    n = FacetNormal(m)
    assert assemble(dot(w, n) * ds) == approx(assemble(div(w) * dx(domain=m)), abs=1e-11)
    assert assemble(dot(w, n) * ds("front")) == approx(1.0)
    assert assemble(dot(w, n) * ds("top")) == approx(2.0)


def test_functional_function_derivatives_on_dx():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 3)
    f = Function(V).interpolate(lambda x: x[0] ** 2 * x[1] + x[1] ** 3)
    assert assemble(f * dx) == approx(1 / 6 + 1 / 4)
    assert assemble(inner(grad(f), grad(f)) * dx) == approx(
        assemble((4 * SpatialCoordinate(m)[0] ** 2 * SpatialCoordinate(m)[1] ** 2 + (SpatialCoordinate(m)[0] ** 2 + 3 * SpatialCoordinate(m)[1] ** 2) ** 2) * dx(domain=m)))
    assert assemble(f.dx(0) * dx) == approx(0.5)
    assert assemble(div(grad(f)) * dx) == approx(assemble(8 * SpatialCoordinate(m)[1] * dx(domain=m)))


def test_functional_returns_float_types():
    m = mk("tri")
    r = assemble(Constant(m, 1.0) * dx)
    assert isinstance(r, float)


def test_functional_conditional_integrand():
    m = UnitSquareMesh(8, 8)
    x = SpatialCoordinate(m)
    assert assemble(conditional(lt(x[0], 0.5), 1.0, 0.0) * dx(domain=m, degree=3)) == approx(0.5)
    assert assemble(conditional(lt(x[0], 0.5), 1.0, 0.0) * ds("bottom")) == approx(0.5)


def test_functional_sum_of_measures():
    m = mk("tri")
    x = SpatialCoordinate(m)
    assert assemble(x[0] * dx + x[0] * ds("right") + x[1] * ds) == approx(0.5 + 1.0 + 2.0)


def test_empty_form_handling():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    v = TestFunction(V)
    with pytest.raises(Exception):
        assemble(0 * v * dx)


def test_assemble_scalar_valued_vector_dot_in_dx():
    m = mk("tri")
    x = SpatialCoordinate(m)
    assert assemble(dot(x, x) * dx) == approx(2 / 3)
    assert assemble(inner(grad(x[0] * x[1]), grad(x[0] * x[1])) * dx) == approx(2 / 3)
    assert assemble(det(grad(x)) * dx) == approx(1.0)


def test_domain_less_integrand_with_domain_argument():
    m = mk("tri")
    assert assemble(1.0 * dx(domain=m)) == approx(1.0)
    assert assemble(Constant(m, 3.0) * dx(m)) == approx(3.0)


def test_assemble_bilinear_flags_pass_through():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    A0 = A(u * v * dx)
    A1 = A(u * v * dx, symmetric=True)
    close(A0, A1)


# ---------- index notation inside forms

@pytest.mark.parametrize("cell", ["tri", "tet", "quad"])
def test_elasticity_index_notation_matches_compact(cell):
    m = mk(cell)
    d = m.dim
    V = VectorFunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    mu, lam = 1.3, 0.7
    eps = lambda w: 0.5 * (grad(w) + grad(w).T)
    compact = inner(2 * mu * eps(u) + lam * tr(eps(u)) * Identity(d), eps(v)) * dx
    e = 0.5 * (u[i].dx(j) + u[j].dx(i))
    ev_ = 0.5 * (v[i].dx(j) + v[j].dx(i))
    delta = Identity(d)
    sig = 2 * mu * e + lam * u[k].dx(k) * delta[i, j]
    idx = sig * ev_ * dx
    close(A(idx), A(compact), 1e-11)
    E = lambda w: ngs.Sym(ngs.Grad(w))
    nat = native_bf(V.fes, lambda U, W: ngs.InnerProduct(2 * mu * E(U) + lam * ngs.Trace(E(U)) * ngs.Id(d), E(W)) * ngs.dx)
    close(A(compact), nat, 1e-11)


def test_index_forms_gradient_contractions():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    base = A(inner(grad(u), grad(v)) * dx)
    close(A(grad(u)[i, j] * grad(v)[i, j] * dx), base)
    close(A(u[i].dx(j) * v[i].dx(j) * dx), base)
    close(A(as_tensor(grad(u)[i, j], (j, i))[k, l] * as_tensor(grad(v)[i, j], (j, i))[k, l] * dx), base)
    close(A(u[i] * v[i] * dx), A(inner(u, v) * dx))
    close(A(u[i].dx(i) * v[j].dx(j) * dx), A(div(u) * div(v) * dx))
    close(A(grad(u)[i, j] * grad(v)[j, i] * dx), A(inner(grad(u), grad(v).T) * dx))
    close(A(grad(u)[0, 1] * grad(v)[1, 0] * dx), A(grad(u)[0, 1] * grad(v)[1, 0] * dx))


def test_index_forms_in_mixed_space():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    compact = inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx
    idx = grad(u)[i, j] * grad(v)[i, j] * dx - v[i].dx(i) * p * dx - q * u[j].dx(j) * dx
    close(A(idx), A(compact))
    w, z = TrialFunction(W), TestFunction(W)
    close(A(w[0] * z[0] * dx + w[1] * z[1] * dx + w[2] * z[2] * dx), A(inner(u, v) * dx + p * q * dx))
    close(A(grad(w)[0, 1] * grad(z)[0, 1] * dx), A(grad(u)[0, 1] * grad(v)[0, 1] * dx))
    close(A(grad(w)[2, 0] * z[2] * dx), A(grad(p)[0] * q * dx))


def test_index_forms_with_coefficient_tensor():
    m = mk("tri")
    V = VectorFunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    K = as_matrix([[1 + x[0], x[1]], [x[1], 2 + x[0] * x[1]]])
    close(A(K[i, j] * u[j].dx(k) * v[i].dx(k) * dx, ), A(inner(K * grad(u), grad(v)) * dx), 1e-11)
    close(A(K[i, j] * u[j] * v[i] * dx), A(dot(K * u, v) * dx), 1e-11)


def test_index_forms_3d_rank2():
    m = mk("tet")
    V = VectorFunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    S = lambda w: as_tensor(0.5 * (w[i].dx(j) + w[j].dx(i)), (i, j))
    close(A(inner(S(u), S(v)) * dx), A(inner(sym(grad(u)), sym(grad(v))) * dx))
    close(A(S(u)[i, j] * S(v)[i, j] * dx), A(inner(sym(grad(u)), sym(grad(v))) * dx))


def test_curl_forms_3d():
    m = mk("tet")
    V = FunctionSpace(m, "N1curl", 1)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(inner(curl(u), curl(v)) * dx), native_bf(V.fes, lambda U, W: ngs.curl(U) * ngs.curl(W) * ngs.dx))


def _side_quadrature(fn, npts=6):
    """sum over the 4 sides of the unit square of fn(point, normal) ds."""
    xs, ws = np.polynomial.legendre.leggauss(npts)
    xs, ws = (xs + 1) / 2, ws / 2
    tot = 0.0
    for p, nrm in [((lambda t: (t, 0.0)), (0, -1)), ((lambda t: (1.0, t)), (1, 0)),
                   ((lambda t: (t, 1.0)), (0, 1)), ((lambda t: (0.0, t)), (-1, 0))]:
        tot += sum(w * fn(p(t), np.array(nrm, float)) for t, w in zip(xs, ws))
    return tot


F_ = lambda p: np.array([p[0] + 2 * p[1], 3 * p[0] - p[1]])
G_ = lambda p: np.array([1 + p[1], p[0] * p[1]])


def test_curl_2d_forms():
    m = mk("tri")
    V = FunctionSpace(m, "N1curl", 2)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(curl(u) * curl(v) * dx), native_bf(V.fes, lambda U, W: ngs.curl(U) * ngs.curl(W) * ngs.dx))
    close(A(inner(u, v) * dx), native_bf(V.fes, lambda U, W: U * W * ngs.dx))


# ---------- H(div) / H(curl) boundary and facet forms

def test_rt_normal_trace_bilinear_form():
    m = mk("tri")
    Rt, Dg = FunctionSpace(m, "RT", 2), FunctionSpace(m, "DG", 1)
    u, q = TrialFunction(Rt), TestFunction(Dg)
    n = FacetNormal(m)
    f = Function(Rt).interpolate(lambda x: (x[0] + 2 * x[1], 3 * x[0] - x[1]))
    g = Function(Dg).interpolate(lambda x: 1 + x[0] + 2 * x[1])
    ref = _side_quadrature(lambda p, nr: (F_(p) @ nr) * (1 + p[0] + 2 * p[1]))
    assert g.array @ A(dot(u, n) * q * ds) @ f.array == approx(ref, rel=1e-11)


def test_rt_normal_normal_ds():
    m = mk("tri")
    Rt = FunctionSpace(m, "RT", 2)
    u, v = TrialFunction(Rt), TestFunction(Rt)
    n = FacetNormal(m)
    f = Function(Rt).interpolate(lambda x: (x[0] + 2 * x[1], 3 * x[0] - x[1]))
    g = Function(Rt).interpolate(lambda x: (1 + x[1], x[0] * x[1]))
    ref = _side_quadrature(lambda p, nr: (F_(p) @ nr) * (G_(p) @ nr))
    assert g.array @ A(dot(u, n) * dot(v, n) * ds) @ f.array == approx(ref, rel=1e-11)
    assert assemble(dot(f, n) * dot(g, n) * ds) == approx(ref, rel=1e-11)


def test_hcurl_tangential_ds():
    m = mk("tri")
    Nc = FunctionSpace(m, "N1curl", 2)
    u, v = TrialFunction(Nc), TestFunction(Nc)
    n = FacetNormal(m)
    t = perp(n)
    f = Function(Nc).interpolate(lambda x: (x[0] + 2 * x[1], 3 * x[0] - x[1]))
    g = Function(Nc).interpolate(lambda x: (1 + x[1], x[0] * x[1]))
    tg = lambda nr: np.array([-nr[1], nr[0]])
    ref = _side_quadrature(lambda p, nr: (F_(p) @ tg(nr)) * (G_(p) @ tg(nr)))
    assert g.array @ A(dot(u, t) * dot(v, t) * ds) @ f.array == approx(ref, rel=1e-11)


def test_rt_jump_facet_form():
    m = mk("tri")
    Rt = FunctionSpace(m, "RT", 1)
    u, v = TrialFunction(Rt), TestFunction(Rt)
    n = FacetNormal(m)
    a = dot(jump(u), n("+")) * dot(jump(v), n("+")) * dS
    fes = Rt.ngs_space(True)
    nn = ngs.specialcf.normal(2)
    nat = native_bf(fes, lambda U, W: (((U - U.Other()) * nn) * ((W - W.Other()) * nn)) * ngs.dx(skeleton=True))
    close(A(a), nat, 1e-11)


def test_rt_normal_continuity_zero_jump_of_interpolant():
    m = mk("tri")
    Rt = FunctionSpace(m, "RT", 2)
    f = Function(Rt).interpolate(lambda x: (x[0] + 2 * x[1], 3 * x[0] - x[1]))
    Q = FunctionSpace(m, "DG", 1)
    q = TestFunction(Q)
    n = FacetNormal(m)
    assert np.abs(vec(jump(f, n) * jump(q) * dS)).max() < 1e-12
    assert np.abs(vec(dot(jump(f), n("+")) * jump(q) * dS)).max() < 1e-12


def test_hcurl_tangential_continuity_of_interpolant():
    m = mk("tri")
    Nc = FunctionSpace(m, "N1curl", 2)
    f = Function(Nc).interpolate(lambda x: (x[1] + x[0] * x[1], x[0] - x[1] ** 2))
    q = TestFunction(FunctionSpace(m, "DG", 1))
    n = FacetNormal(m)
    assert np.abs(vec(dot(jump(f), perp(n("+"))) * jump(q) * dS)).max() < 1e-12
    assert np.abs(vec(dot(jump(f), n("+")) * jump(q) * dS)).max() > 1e-6


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex"])
def test_interior_normal_orientation_antisymmetric(cell):
    m = mk(cell)
    V = FunctionSpace(m, "DG", 0)
    q = TestFunction(V)
    n = FacetNormal(m)
    assert np.abs(vec(dot(n("+"), n("-")) * q("+") * dS) + vec(q("+") * dS)).max() < 1e-12
    assert np.abs(vec((n("+")[0] + n("-")[0]) * q("+") * dS)).max() < 1e-12


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex"])
def test_divergence_theorem_on_dg_with_facets(cell):
    m = mk(cell)
    d = m.dim
    V = VectorFunctionSpace(m, "DG", 2)
    w = Function(V).interpolate(lambda x: tuple([x[0] ** 2 + x[1], x[1] * x[0], x[0] + 2][:d]))
    n = FacetNormal(m)
    lhs_ = assemble(div(w) * dx(domain=m))
    bnd = assemble(dot(w, n) * ds)
    jumps = assemble(dot(jump(w), n("+")) * dS(domain=m))
    assert lhs_ == approx(bnd - jumps, rel=1e-10, abs=1e-12)
    exact = assemble(div(as_vector([SpatialCoordinate(m)[0] ** 2 + SpatialCoordinate(m)[1], SpatialCoordinate(m)[1] * SpatialCoordinate(m)[0],
                                    SpatialCoordinate(m)[0] + 2][:d])) * dx(domain=m))
    assert exact == approx(bnd, rel=1e-10)


def test_facet_functional_h1_has_no_jump_of_value_and_averages():
    m = mk("tri")
    VC = FunctionSpace(m, "Lagrange", 2)
    f = Function(VC).interpolate(lambda x: x[0] ** 2 * x[1])
    assert assemble(avg(f) * dS) == approx(assemble(f("+") * dS), rel=1e-12)
    assert assemble(f("-") * dS) == approx(assemble(f("+") * dS), rel=1e-12)
    n = FacetNormal(m)
    assert assemble(jump(grad(f), n) ** 2 * dS) > 1e-6


def test_sign_elem_ops_diag_dn():
    m = mk("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] * x[1])
    x = SpatialCoordinate(m)
    p = (0.3, 0.4)
    from conftest import ev
    val = lambda e: ev(to_cf(e), m, p)
    assert val(sign(x[0] - 0.5)) == approx(-1.0)
    assert val(elem_mult(as_vector((x[0], 2.0)), as_vector((3.0, x[1])))) == approx([0.9, 0.8])
    assert val(elem_div(as_vector((x[0], 2.0)), as_vector((3.0, x[1])))) == approx([0.1, 5.0])
    assert val(elem_pow(as_vector((x[0], 2.0)), as_vector((3.0, x[1])))) == approx([0.027, 2 ** 0.4])
    assert val(elem_op(sin, as_vector((x[0], 2.0)))) == approx([math.sin(0.3), math.sin(2.0)])
    assert val(diag(as_matrix(((1.0, 2.0), (3.0, x[0]))))) == approx([1, 0, 0, 0.3])
    assert val(diag_vector(as_matrix(((1.0, 2.0), (3.0, x[0]))))) == approx([1, 0.3])
    assert val(rot(as_vector((-x[1], x[0])))) == approx(2.0)
    n = FacetNormal(m)
    assert assemble(Dn(f) * ds("right")) == approx(assemble(f.dx(0) * ds("right")))


def test_jacobian_quantities():
    m = mk("tri")
    p = m.ngs(0.3, 0.4)
    J = np.array(to_cf(Jacobian(m))(p)).reshape(2, 2)
    assert abs(to_cf(JacobianDeterminant(m))(p)) == approx(abs(np.linalg.det(J)))
    assert np.array(to_cf(JacobianInverse(m))(p)).reshape(2, 2) == approx(np.linalg.inv(J))
    # |det J| is twice the triangle area
    assert assemble(abs(JacobianDeterminant(m)) * dx) / 2 == approx(assemble(CellVolume(m) * dx))


@pytest.mark.xfail(reason="FacetArea is not translated", strict=True)
def test_facet_area():
    m = mk("tri")
    to_cf(FacetArea(m))


@pytest.mark.xfail(reason="div(div(sigma)) of an HHJ function is not implemented (second derivatives of tensor elements)", strict=True)
def test_hhj_divdiv_form():
    m = mk("tri")
    V = FunctionSpace(m, "HHJ", 1)
    s, t = TrialFunction(V), TestFunction(V)
    assemble(div(div(s)) * div(div(t)) * dx)


def test_dS_1d_mesh():
    m = UnitIntervalMesh(3)
    V = FunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    close(A(jump(u) * jump(v) * dS), dgnat(V, lambda U, W: (U - U.Other()) * (W - W.Other()) * ngs.dx(skeleton=True)))
    assert assemble(Constant(m, 1.0) * dS(domain=m)) == approx(2.0)


def _native_ds_lf(V, name, d):
    fes = V.fes
    lf = ngs.LinearForm(fes)
    lf += ngs.x ** 4 * fes.TestFunction() * ngs.ds(name, intrules={ngs.ET.SEGM: ngs.IntegrationRule(ngs.ET.SEGM, d)})
    lf.Assemble()
    return lf.vec.FV().NumPy().copy()


@pytest.mark.parametrize("d", [2, 4, 8])
def test_ds_degree_matches_native_rule(d):
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "Lagrange", 1)
    x = SpatialCoordinate(m)
    close(vec(x[0] ** 4 * TestFunction(V) * ds("bottom", degree=d)), _native_ds_lf(V, "bottom", d), 1e-12)


@pytest.mark.parametrize("d", [0, 1])
def test_ds_low_degree_respected(d):
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "Lagrange", 1)
    x = SpatialCoordinate(m)
    close(vec(x[0] ** 4 * TestFunction(V) * ds("bottom", degree=d)), _native_ds_lf(V, "bottom", d), 1e-12)


def test_dx_degree_in_form_matches_native_rule():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "Lagrange", 1)
    x = SpatialCoordinate(m)
    for d in (2, 4, 8):
        lf = ngs.LinearForm(V.fes)
        lf += ngs.x ** 4 * V.fes.TestFunction() * ngs.dx(intrules={ngs.ET.TRIG: ngs.IntegrationRule(ngs.ET.TRIG, d)})
        lf.Assemble()
        close(vec(x[0] ** 4 * TestFunction(V) * dx(degree=d)), lf.vec.FV().NumPy(), 1e-12)


def test_mixed_degrees_in_one_form():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    a = x[0] ** 3 * u * v * dx(degree=1) + x[1] ** 3 * u * v * dx(degree=6)
    close(A(a), A(x[0] ** 3 * u * v * dx(degree=1)) + A(x[1] ** 3 * u * v * dx(degree=6)))


def test_curved_mesh_piola_and_geometry():
    from netgen.occ import Circle, OCCGeometry
    disk = Circle((0, 0), 1).Face()
    disk.edges.name = "rim"
    mc = Mesh(OCCGeometry(disk, dim=2).GenerateMesh(maxh=0.4)).curve(3)
    for fam in ("RT", "N1curl"):
        V = FunctionSpace(mc, fam, 2)
        if fam == "RT":
            f = Function(V).interpolate(lambda x: (x[0], x[1]))
            assert assemble(div(f) * dx(domain=mc)) == approx(2 * math.pi, rel=1e-3)
            assert assemble(dot(f, FacetNormal(mc)) * ds) == approx(2 * math.pi, rel=1e-3)
        else:
            f = Function(V).interpolate(lambda x: (-x[1], x[0]))
            assert assemble(curl(f) * dx(domain=mc)) == approx(2 * math.pi, rel=1e-3)
    W = FunctionSpace(mc, "Lagrange", 3)
    g = Function(W).interpolate(lambda x: x[0] ** 2 + x[1] ** 2)
    assert assemble(inner(grad(g), grad(g)) * dx(domain=mc)) == approx(2 * math.pi, rel=1e-3)
    assert assemble(Constant(mc, 1.0) * dx) == approx(math.pi, rel=1e-3)


def _mixed_dg():
    m = mk("tri")
    W = VectorFunctionSpace(m, "DG", 1) * FunctionSpace(m, "DG", 1)
    return m, W


def test_mixed_dg_dS_bilinear():
    m, W = _mixed_dg()
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    M = A(jump(p) * jump(q) * dS + dot(avg(u), FacetNormal(m)("+")) * jump(q) * dS)
    assert M.shape == (W.dim(), W.dim())


def test_mixed_dg_dS_linear():
    m, W = _mixed_dg()
    v, q = TestFunctions(W)
    L = vec(jump(q) * dS + dot(avg(v), FacetNormal(m)("+")) * dS)
    assert L.shape == (W.dim(),)
    x = SpatialCoordinate(m)
    n = FacetNormal(m)
    c = Constant(m, (1.0, 2.0))
    V1, Q1 = FunctionSpace(m, "DG", 1), FunctionSpace(m, "DG", 1)
    vv, qq = TestFunction(FunctionSpace(m, "DG", 1)), TestFunction(Q1)
    Lq = vec(x[0] * jump(q) * dS)
    nat_q = vec(x[0] * jump(qq) * dS)
    nV = VectorFunctionSpace(m, "DG", 1).dim()
    close(Lq[nV:], nat_q)
    assert np.abs(Lq[:nV]).max() == 0


def test_mixed_dg_dS_solve():
    m, W = _mixed_dg()
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    w = Function(W)
    solve(jump(p) * jump(q) * dS + p * q * dx + dot(u, v) * dx == q * dx, w)

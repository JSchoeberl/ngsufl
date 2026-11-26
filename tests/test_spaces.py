import math

import numpy as np
import ngsolve as ngs
import pytest

from ngsufl import *
from conftest import approx

_m = {}


def mesh(name):
    if name not in _m:
        _m[name] = {
            "seg": lambda: UnitIntervalMesh(4),
            "tri": lambda: UnitSquareMesh(2, 2),
            "quad": lambda: UnitSquareMesh(2, 2, quadrilateral=True),
            "tet": lambda: BoxMesh(2, 1, 1),
            "hex": lambda: BoxMesh(2, 1, 1, hexahedral=True),
        }[name]()
    return _m[name]


def counts(m):
    g = m.ngs
    nface = g.nface if m.dim == 3 else g.ne
    return g.nv, g.nedge, nface, g.ne


def C(n, k):
    return math.comb(n, k) if n >= k >= 0 else 0


def lagrange(cell, k, nv, ne_, nf, nc):
    if cell == "seg":
        return nv + (k - 1) * nc
    if cell == "tri":
        return nv + (k - 1) * ne_ + C(k - 1, 2) * nc
    if cell == "quad":
        return nv + (k - 1) * ne_ + (k - 1) ** 2 * nc
    if cell == "tet":
        return nv + (k - 1) * ne_ + C(k - 1, 2) * nf + C(k - 1, 3) * nc
    return nv + (k - 1) * ne_ + (k - 1) ** 2 * nf + (k - 1) ** 3 * nc


def dg(cell, k, nc):
    n = {"seg": k + 1, "tri": (k + 1) * (k + 2) // 2, "quad": (k + 1) ** 2,
         "tet": (k + 1) * (k + 2) * (k + 3) // 6, "hex": (k + 1) ** 3}[cell]
    return n * nc


@pytest.mark.parametrize("cell", ["seg", "tri", "quad", "tet", "hex"])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_lagrange_dg_ndof(cell, k):
    m = mesh(cell)
    nv, ne_, nf, nc = counts(m)
    assert FunctionSpace(m, "Lagrange", k).dim() == lagrange(cell, k, nv, ne_, nf, nc)
    assert FunctionSpace(m, "DG", k).dim() == dg(cell, k, nc)
    assert FunctionSpace(m, "DG", k - 1).dim() == dg(cell, k - 1, nc)


@pytest.mark.parametrize("name", ["P", "CG", "Q", "Lagrange"])
def test_lagrange_aliases(name):
    m = mesh("tri")
    assert FunctionSpace(m, name, 2).dim() == FunctionSpace(m, "Lagrange", 2).dim()


@pytest.mark.parametrize("name", ["DG", "DP", "DQ"])
def test_dg_aliases(name):
    m = mesh("tri")
    assert FunctionSpace(m, name, 2).dim() == dg("tri", 2, m.ngs.ne)


@pytest.mark.parametrize("cell", ["tri", "quad", "tet", "hex"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_vector_tensor_ndof(cell, k):
    m = mesh(cell)
    d = m.dim
    s = FunctionSpace(m, "Lagrange", k).dim()
    assert VectorFunctionSpace(m, "Lagrange", k).dim() == d * s
    assert TensorFunctionSpace(m, "Lagrange", k).dim() == d * d * s
    sd = FunctionSpace(m, "DG", k).dim()
    assert VectorFunctionSpace(m, "DG", k).dim() == d * sd


def test_cr_real():
    m = mesh("tri")
    assert FunctionSpace(m, "CR", 1).dim() == m.ngs.nedge
    m3 = mesh("tet")
    assert FunctionSpace(m3, "CR", 1).dim() == m3.ngs.nface
    assert FunctionSpace(m, "Real", 0).dim() == 1
    assert FunctionSpace(m, "R", 0).dim() == 1


def rt(cell, k, nf, nc, nedge):
    if cell == "tri":
        return k * nedge + k * (k - 1) * nc
    return nf * k * (k + 1) // 2 + nc * (k - 1) * k * (k + 1) // 2


def bdm(cell, k, nf, nc, nedge):
    if cell == "tri":
        return nedge * (k + 1) + nc * (k * k - 1)
    return nf * (k + 1) * (k + 2) // 2 + nc * (k + 1) * (k + 2) * (k - 1) // 2


def n1(cell, k, nf, nc, nedge):
    if cell == "tri":
        return k * nedge + k * (k - 1) * nc
    return k * nedge + nf * k * (k - 1) + nc * k * (k - 1) * (k - 2) // 2


def n2(cell, k, nf, nc, nedge):
    if cell == "tri":
        return nedge * (k + 1) + nc * (k + 1) * (k - 1)
    return nedge * (k + 1) + nf * (k + 1) * (k - 1) + nc * (k + 1) * (k - 1) * (k - 2) // 2


@pytest.mark.parametrize("cell", ["tri", "tet"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_hdiv_hcurl_ndof(cell, k):
    m = mesh(cell)
    nv, ne_, nf, nc = counts(m)
    assert FunctionSpace(m, "RT", k).dim() == rt(cell, k, nf, nc, ne_)
    assert FunctionSpace(m, "BDM", k).dim() == bdm(cell, k, nf, nc, ne_)
    assert FunctionSpace(m, "N1curl", k).dim() == n1(cell, k, nf, nc, ne_)
    assert FunctionSpace(m, "N2curl", k).dim() == n2(cell, k, nf, nc, ne_)


@pytest.mark.parametrize("alias,canon", [("N1F", "RT"), ("Raviart-Thomas", "RT"), ("N2F", "BDM"), ("N1E", "N1curl"),
                                          ("N2E", "N2curl"), ("Nedelec 1st kind H(curl)", "N1curl")])
def test_family_aliases(alias, canon):
    m = mesh("tri")
    assert FunctionSpace(m, alias, 2).dim() == FunctionSpace(m, canon, 2).dim()


@pytest.mark.parametrize("fam", ["Regge", "HHJ"])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_regge_hhj_ndof(fam, k):
    m = mesh("tri")
    nv, ne_, nf, nc = counts(m)
    want = ne_ * (k + 1) + nc * 3 * k * (k + 1) // 2
    assert FunctionSpace(m, fam, k).dim() == want


def test_unknown_family_and_bad_degree():
    m = mesh("tri")
    with pytest.raises(ValueError):
        FunctionSpace(m, "Bogus", 1)
    with pytest.raises(ValueError):
        FunctionSpace(m, "Lagrange", 0)
    with pytest.raises(ValueError):
        FunctionSpace(m, "RT", 0)
    with pytest.raises(ValueError):
        FunctionSpace(m, "CR", 2)


def test_dolfinx_style_tuple_spec():
    m = mesh("tri")
    assert FunctionSpace(m, ("Lagrange", 2)).dim() == FunctionSpace(m, "Lagrange", 2).dim()
    assert FunctionSpace(m, ("Lagrange", 2, (2,))).dim() == VectorFunctionSpace(m, "Lagrange", 2).dim()
    assert FunctionSpace(m, ("Lagrange", 1, (2, 2))).dim() == TensorFunctionSpace(m, "Lagrange", 1).dim()


def test_shapes():
    m = mesh("tri")
    V = FunctionSpace(m, "Lagrange", 1)
    Vv = VectorFunctionSpace(m, "Lagrange", 1)
    Vt = TensorFunctionSpace(m, "Lagrange", 1)
    assert Function(V).ufl_shape == ()
    assert Function(Vv).ufl_shape == (2,)
    assert Function(Vt).ufl_shape == (2, 2)
    assert FunctionSpace(m, "RT", 1).value_shape == (2,)
    assert FunctionSpace(m, "N1curl", 1).value_shape == (2,)
    assert FunctionSpace(m, "Regge", 1).value_shape == (2, 2)
    assert FunctionSpace(m, "HHJ", 1).value_shape == (2, 2)
    m3 = mesh("tet")
    assert FunctionSpace(m3, "RT", 1).value_shape == (3,)
    assert VectorFunctionSpace(m3, "Lagrange", 1).value_shape == (3,)
    assert TensorFunctionSpace(m3, "DG", 1).value_shape == (3, 3)
    assert TrialFunction(Vv).ufl_shape == (2,)
    assert grad(Function(Vv)).ufl_shape == (2, 2)


def test_vector_space_custom_dimension():
    m = mesh("tri")
    V = VectorFunctionSpace(m, "Lagrange", 1, dim=3)
    assert V.value_shape == (3,)
    assert V.dim() == 3 * FunctionSpace(m, "Lagrange", 1).dim()


def test_tensor_space_custom_shape():
    m = mesh("tri")
    V = TensorFunctionSpace(m, "Lagrange", 1, shape=(3, 3))
    assert V.value_shape == (3, 3)


@pytest.mark.xfail(reason="non-square tensor block shapes are not implemented", strict=True)
def test_nonsquare_tensor_space():
    m = mesh("tri")
    V = TensorFunctionSpace(m, "Lagrange", 1, shape=(2, 3))
    assert V.dim() == 6 * FunctionSpace(m, "Lagrange", 1).dim()


def test_degree_2_tensor_of_vector_dg():
    m = mesh("tet")
    assert TensorFunctionSpace(m, "Lagrange", 2).dim() == 9 * FunctionSpace(m, "Lagrange", 2).dim()


# ---------- mixed spaces and sub spaces

def test_mixed_ndof_and_subspaces():
    m = mesh("tri")
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    assert W.dim() == V.dim() + Q.dim()
    assert W.num_sub_spaces() == 2
    assert W.sub(0).dof_range() == range(0, V.dim())
    assert W.sub(1).dof_range() == range(V.dim(), V.dim() + Q.dim())
    n = Q.dim() * 0 + FunctionSpace(m, "Lagrange", 2).dim()
    assert V.sub(0).dof_range() == range(0, n)
    assert V.sub(1).dof_range() == range(n, 2 * n)
    assert W.sub(0).sub(1).dof_range() == range(n, 2 * n)
    assert len(W.sub(0).dof_range()) == V.dim()


def test_mixed_function_space_function_and_three_way():
    m = mesh("tri")
    V = FunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "DG", 1)
    R = FunctionSpace(m, "Real", 0)
    W = MixedFunctionSpace([V, Q, R])
    assert W.dim() == V.dim() + Q.dim() + 1
    assert W.sub(2).dof_range() == range(V.dim() + Q.dim(), V.dim() + Q.dim() + 1)
    W2 = V * Q * R
    assert W2.dim() == W.dim()
    assert W2.num_sub_spaces() == 3


def test_mixed_with_vector_flattens_components():
    m = mesh("tri")
    V = VectorFunctionSpace(m, "Lagrange", 1)
    W = V * FunctionSpace(m, "Lagrange", 1)
    w = Function(W)
    assert w.ufl_shape == (3,)
    u, p = split(w)
    assert u.ufl_shape == (2,)
    assert p.ufl_shape == ()


def test_split_trial_test_functions():
    m = mesh("tri")
    W = VectorFunctionSpace(m, "Lagrange", 2) * FunctionSpace(m, "Lagrange", 1)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    assert u.ufl_shape == (2,) and p.ufl_shape == () and v.ufl_shape == (2,)
    w = Function(W)
    w.sub(1).interpolate(lambda x: x[0])
    assert w.sub(1).at(0.3, 0.2) == approx(0.3)


def test_mixed_function_at():
    m = mesh("tri")
    W = VectorFunctionSpace(m, "Lagrange", 2) * FunctionSpace(m, "Lagrange", 1)
    w = Function(W)
    w.sub(1).interpolate(lambda x: x[0])
    assert w.at(0.3, 0.2)[2] == approx(0.3)


def test_mixed_rt_dg():
    m = mesh("tri")
    W = FunctionSpace(m, "RT", 2) * FunctionSpace(m, "DG", 1)
    nv, ne_, nf, nc = counts(m)
    assert W.dim() == rt("tri", 2, nf, nc, ne_) + dg("tri", 1, nc)
    w = Function(W)
    assert w.ufl_shape == (3,)
    s, u = w.subfunctions
    assert s.ufl_shape == (2,)


def test_subfunctions_share_data():
    m = mesh("tri")
    V = VectorFunctionSpace(m, "Lagrange", 1)
    Q = FunctionSpace(m, "Lagrange", 1)
    w = Function(V * Q)
    u, p = w.subfunctions
    p.interpolate(1.0)
    u.interpolate((2.0, 3.0))
    assert u.at(0.3, 0.3) == approx([2.0, 3.0])
    assert p.at(0.3, 0.3) == approx(1.0)
    w.array[:] = 0.0
    assert u.at(0.3, 0.3) == approx([0.0, 0.0])
    assert p.at(0.3, 0.3) == approx(0.0)
    assert w.sub(0) is u
    assert w.split()[1] is p


def test_subfunction_array_is_view():
    m = mesh("tri")
    W = FunctionSpace(m, "Lagrange", 1) * FunctionSpace(m, "Lagrange", 1)
    w = Function(W)
    a, b = w.subfunctions
    a.array[:] = 1.0
    assert w.array[: a.function_space().dim()].tolist() == [1.0] * a.function_space().dim()
    assert np.all(w.array[a.function_space().dim():] == 0.0)


def test_mixed_interpolate_tuple():
    m = mesh("tri")
    W = FunctionSpace(m, "Lagrange", 1) * VectorFunctionSpace(m, "Lagrange", 2)
    w = Function(W).interpolate(lambda x: (x[0], 2.0, x[0] * x[1]))
    assert w.at(0.3, 0.4) == approx((0.3, 2.0, 0.12))
    assert w.sub(1).at(0.3, 0.4) == approx((2.0, 0.12))


def test_mixed_assign_from_other():
    m = mesh("tri")
    W = FunctionSpace(m, "Lagrange", 1) * FunctionSpace(m, "DG", 1)
    a, b = Function(W), Function(W)
    a.sub(0).interpolate(2.0)
    a.sub(1).interpolate(3.0)
    b.assign(a)
    assert b.sub(0).at(0.4, 0.4) == approx(2.0) and b.sub(1).at(0.4, 0.4) == approx(3.0)
    a.sub(0).interpolate(5.0)
    assert b.sub(0).at(0.4, 0.4) == approx(2.0)


def test_nested_mixed_flattened():
    m = mesh("tri")
    V, Q = FunctionSpace(m, "Lagrange", 1), FunctionSpace(m, "DG", 0)
    W = (V * Q) * V
    assert W.num_sub_spaces() == 3
    assert W.dim() == 2 * V.dim() + Q.dim()


# ---------- Function API

@pytest.fixture(scope="module")
def tri():
    return mesh("tri")


def test_interpolate_kinds(tri):
    V = FunctionSpace(tri, "Lagrange", 2)
    x = SpatialCoordinate(tri)
    f = Function(V)
    p = (0.3, 0.4)
    f.interpolate(2.5)
    assert f.at(*p) == approx(2.5)
    f.interpolate(x[0] * x[1])
    assert f.at(*p) == approx(0.12)
    f.interpolate(lambda x: x[0] ** 2 + np.sin(x[1]) * 0)
    assert f.at(*p) == approx(0.09)
    f.interpolate(ngs.x + 2 * ngs.y)
    assert f.at(*p) == approx(1.1)
    f.interpolate(Constant(tri, 7.0))
    assert f.at(*p) == approx(7.0)
    g = Function(V).interpolate(lambda x: 1 + x[0])
    f.interpolate(2 * g)
    assert f.at(*p) == approx(2.6)
    assert f.interpolate(1.0) is f


def test_interpolate_vector_kinds(tri):
    V = VectorFunctionSpace(tri, "Lagrange", 2)
    f = Function(V)
    p = (0.3, 0.4)
    f.interpolate((1.0, 2.0))
    assert f.at(*p) == approx([1.0, 2.0])
    f.interpolate([x_ for x_ in (3.0, 4.0)])
    assert f.at(*p) == approx([3.0, 4.0])
    f.interpolate(lambda x: (x[0], x[1] ** 2))
    assert f.at(*p) == approx([0.3, 0.16])
    f.interpolate(as_vector((SpatialCoordinate(tri)[1], 1.0)))
    assert f.at(*p) == approx([0.4, 1.0])
    f.interpolate(ngs.CF((ngs.y, ngs.x)))
    assert f.at(*p) == approx([0.4, 0.3])
    f.interpolate(2.0)
    assert f.at(*p) == approx([2.0, 2.0])


def test_interpolate_callable_returns_ndarray(tri):
    V = VectorFunctionSpace(tri, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: np.array([x[0], x[1]]))
    assert f.at(0.3, 0.4) == approx([0.3, 0.4])


def test_interpolate_tensor(tri):
    V = TensorFunctionSpace(tri, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: ((x[0], 1.0), (2.0, x[1])))
    assert f.at(0.3, 0.4) == approx([0.3, 1.0, 2.0, 0.4])
    f.interpolate(((1, 2), (3, 4)))
    assert f.at(0.3, 0.4) == approx([1, 2, 3, 4])
    f.interpolate(np.eye(2))
    assert f.at(0.3, 0.4) == approx([1, 0, 0, 1])


def test_interpolate_numpy_scalar_and_int(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    f = Function(V)
    f.interpolate(np.float64(1.5))
    assert f.at(0.3, 0.4) == approx(1.5)
    f.interpolate(3)
    assert f.at(0.3, 0.4) == approx(3.0)


def test_interpolate_callable_numpy_functions(tri):
    V = FunctionSpace(tri, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: np.sin(np.pi * x[0]) * np.exp(x[1]) + np.sqrt(1 + x[0] * x[1]) + np.cos(x[0]) + np.arctan(x[1]))
    p = (0.3, 0.4)
    exact = math.sin(math.pi * .3) * math.exp(.4) + math.sqrt(1.12) + math.cos(.3) + math.atan(.4)
    assert f.at(*p) == approx(exact, abs=5e-2)
    assert np.isfinite(f.array).all()


def test_interpolate_callable_with_numpy_constant_array(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: x[0] * np.pi + 2 * np.ones_like(x[0]) * 0 + 1.0)
    assert f.at(0.3, 0.4) == approx(0.3 * math.pi + 1.0)


def test_interpolate_callable_np_where(tri):
    V = FunctionSpace(tri, "DG", 0)
    f = Function(V).interpolate(lambda x: np.where(x[0] < 0.5, 1.0, 0.0))
    assert f.at(0.1, 0.1) == approx(1.0)
    assert f.at(0.9, 0.1) == approx(0.0)


def test_interpolate_callable_comparison(tri):
    V = FunctionSpace(tri, "DG", 0)
    f = Function(V).interpolate(lambda x: (x[0] < 0.5) * 1.0)
    assert f.at(0.1, 0.1) == approx(1.0)


def test_interpolate_is_exact_for_polynomials(tri):
    for k in (1, 2, 3, 4):
        V = FunctionSpace(tri, "Lagrange", k)
        poly = lambda x: sum((0.3 + i) * x[0] ** i * x[1] ** (k - i) for i in range(k + 1))
        f = Function(V).interpolate(poly)
        for p in [(0.13, 0.27), (0.61, 0.33), (0.9, 0.05)]:
            assert f.at(*p) == approx(poly(np.array(p)), rel=1e-10)


def test_interpolate_dg_exact_polynomials(tri):
    V = FunctionSpace(tri, "DG", 3)
    f = Function(V).interpolate(lambda x: x[0] ** 3 - x[1] ** 2 * x[0])
    assert f.at(0.35, 0.55) == approx(0.35 ** 3 - 0.55 ** 2 * 0.35, rel=1e-12)


def test_interpolate_hdiv_hcurl(tri):
    Vr = FunctionSpace(tri, "RT", 2)
    f = Function(Vr).interpolate(lambda x: (x[0] + 2 * x[1], 3 * x[0]))
    assert f.at(0.3, 0.4) == approx([1.1, 0.9])
    Vn = FunctionSpace(tri, "N1curl", 2)
    g = Function(Vn).interpolate(lambda x: (x[1], -x[0] + 1.0))
    assert g.at(0.3, 0.4) == approx([0.4, 0.7])


def test_interpolate_quad_hex():
    for cell in ("quad", "hex"):
        m = mesh(cell)
        V = FunctionSpace(m, "Lagrange", 2)
        f = Function(V).interpolate(lambda x: x[0] * x[1] + x[0] ** 2)
        assert f.at(*(0.3, 0.4, 0.1)[: m.dim]) == approx(0.12 + 0.09)


def test_assign_and_copy(tri):
    V = FunctionSpace(tri, "Lagrange", 2)
    a = Function(V).interpolate(lambda x: x[0])
    b = Function(V)
    assert b.assign(a) is b
    assert b.at(0.3, 0.4) == approx(0.3)
    a.interpolate(5.0)
    assert b.at(0.3, 0.4) == approx(0.3)
    b.assign(0)
    assert np.abs(b.array).max() == 0.0
    b.assign(a)
    b.assign(2.0)
    assert b.at(0.3, 0.4) == approx(2.0)
    b.assign(lambda x: x[1])
    assert b.at(0.3, 0.4) == approx(0.4)
    c = a.copy()
    c.interpolate(1.0)
    assert a.at(0.3, 0.4) == approx(5.0)
    assert c.at(0.3, 0.4) == approx(1.0)
    assert c.function_space() is V


def test_assign_expression_and_arithmetic(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    a = Function(V).interpolate(lambda x: x[0])
    b = Function(V).interpolate(lambda x: x[1])
    c = Function(V).assign(a + 2 * b)
    assert c.at(0.3, 0.4) == approx(1.1)


def test_at_forms(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: 1 + x[0] + 2 * x[1])
    assert f.at(0.3, 0.4) == approx(2.1)
    assert f.at((0.3, 0.4)) == approx(2.1)
    assert f.at(np.array([0.3, 0.4])) == approx(2.1)
    assert f((0.3, 0.4)) == approx(2.1)
    assert f(0.3, 0.4) == approx(2.1)
    assert f.at(0.3, 0.4) == f(0.3, 0.4)


def test_at_outside_mesh_raises(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    f = Function(V)
    with pytest.raises(Exception):
        f.at(5.0, 5.0)


def test_function_restrictions_still_work(tri):
    V = FunctionSpace(tri, "DG", 1)
    f = Function(V)
    assert f("+").ufl_shape == ()
    assert f("-").ufl_shape == ()


def test_function_names_and_vec(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    f = Function(V, name="temperature")
    assert f.name == "temperature"
    assert len(f.vec) == V.dim()
    assert len(f.array) == V.dim()
    f.array[:] = 3.0
    assert np.allclose(f.vec.FV().NumPy(), 3.0)
    assert f.gf.vec.FV().NumPy()[0] == 3.0
    assert f.function_space() is V


def test_function_space_equality_and_hash(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    f, g = Function(V), Function(V)
    assert f.function_space() is g.function_space()
    assert hash(V.ufl_element()) == hash(FunctionSpace(tri, "Lagrange", 1).ufl_element())


def test_function_nodal_values_p1(tri):
    V = FunctionSpace(tri, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: 3 * x[0] + x[1])
    for v in tri.ngs.vertices:
        px, py = v.point
        assert f.array[v.nr] == approx(3 * px + py)


# ---------- Constants

def test_constant_kinds(tri):
    c = Constant(tri, 2.0)
    assert c.ufl_shape == () and c.value == 2.0 and float(c) == 2.0
    cv = Constant(tri, (1.0, 2.0, 3.0))
    assert cv.ufl_shape == (3,)
    assert np.allclose(cv.value, [1, 2, 3])
    ct = Constant(tri, [[1.0, 2.0], [3.0, 4.0]])
    assert ct.ufl_shape == (2, 2)
    assert np.allclose(ct.value, [[1, 2], [3, 4]])
    c1 = Constant(5)
    assert c1.value == 5.0


def test_constant_assign(tri):
    cv = Constant(tri, (1.0, 2.0))
    assert cv.assign((3.0, 4.0)) is cv
    assert np.allclose(cv.value, [3, 4])
    ct = Constant(tri, np.eye(2))
    ct.assign(2 * np.eye(2))
    assert np.allclose(ct.value, 2 * np.eye(2))
    c = Constant(tri, 1.0)
    c.assign(7)
    assert c.value == 7.0


def test_constant_in_expression_and_default_mesh():
    m = UnitSquareMesh(2, 2)
    c = Constant(2.0)
    assert assemble(c * dx(domain=m)) == approx(2.0)
    assert assemble(c * dx(m)) == approx(2.0)


def test_constant_with_domain_keyword(tri):
    c = Constant(3.0, domain=tri)
    assert assemble(c * dx(domain=tri)) == approx(3.0)


def test_constant_wrong_assign_size_raises(tri):
    cv = Constant(tri, (1.0, 2.0))
    with pytest.raises(Exception):
        cv.assign((1.0, 2.0, 3.0))


# ---------- mesh generators

@pytest.mark.parametrize("quad", [False, True])
def test_unit_square_boundaries(quad):
    m = UnitSquareMesh(3, 2, quadrilateral=quad)
    assert m.dim == 2
    assert m.boundaries[:4] == ("bottom", "right", "top", "left")
    one = Constant(m, 1.0)
    assert assemble(one * ds("bottom")) == approx(1.0)
    assert assemble(one * ds("right")) == approx(1.0)
    assert assemble(one * ds) == approx(4.0)
    assert assemble(one * dx(domain=m)) == approx(1.0)
    assert m.ngs.ne == (6 if quad else 12)
    assert m.num_cells == m.ngs.ne
    assert m.num_vertices == m.ngs.nv == 12


def test_boundary_geometry_unit_square():
    m = UnitSquareMesh(3, 3)
    x = SpatialCoordinate(m)
    assert assemble(x[1] * ds("bottom")) == approx(0.0, abs=1e-14)
    assert assemble(x[1] * ds("top")) == approx(1.0)
    assert assemble(x[0] * ds("left")) == approx(0.0, abs=1e-14)
    assert assemble(x[0] * ds("right")) == approx(1.0)


def test_rectangle_mesh_and_origin():
    m = RectangleMesh(4, 3, 1.0, 3.5, originX=-1.0, originY=3.0)
    one = Constant(m, 1.0)
    x = SpatialCoordinate(m)
    assert assemble(one * dx(domain=m)) == approx(1.0)
    assert assemble(one * ds("bottom")) == approx(2.0)
    assert assemble(one * ds("right")) == approx(0.5)
    assert assemble(x[0] * dx(domain=m)) == approx(0.0, abs=1e-14)
    assert assemble(x[1] * ds("top")) == approx(3.5 * 2.0)
    assert assemble(x[0] * ds("left")) == approx(-0.5)


def test_interval_mesh():
    m = IntervalMesh(5, -1.0, 3.0)
    assert m.dim == 1
    assert set(m.boundaries) == {"left", "right"}
    one = Constant(m, 1.0)
    assert assemble(one * dx(domain=m)) == approx(4.0)
    x = SpatialCoordinate(m)
    assert assemble(x[0] * ds("left")) == approx(-1.0)
    assert assemble(x[0] * ds("right")) == approx(3.0)
    assert assemble(x[0] * dx(domain=m)) == approx(4.0)
    assert UnitIntervalMesh(3).ngs.ne == 3


@pytest.mark.parametrize("hexa", [False, True])
def test_box_mesh(hexa):
    m = BoxMesh(2, 3, 1, 2.0, 1.5, 0.5, hexahedral=hexa, origin=(1.0, 0.0, -1.0))
    one = Constant(m, 1.0)
    assert assemble(one * dx(domain=m)) == approx(1.5)
    areas = {"back": 0.75, "front": 0.75, "left": 1.0, "right": 1.0, "bottom": 3.0, "top": 3.0}
    assert set(m.boundaries) == set(areas)
    for nm, a in areas.items():
        assert assemble(one * ds(nm)) == approx(a), nm
    x = SpatialCoordinate(m)
    assert assemble(x[0] * ds("back")) == approx(1.0 * 0.75)
    assert assemble(x[0] * ds("front")) == approx(3.0 * 0.75)
    assert assemble(x[1] * ds("left")) == approx(0.0, abs=1e-14)
    assert assemble(x[1] * ds("right")) == approx(1.5 * 1.0)
    assert assemble(x[2] * ds("bottom")) == approx(-1.0 * 3.0)
    assert assemble(x[2] * ds("top")) == approx(-0.5 * 3.0)
    assert m.ngs.ne == (6 if hexa else 36)


def test_unit_cube_mesh():
    m = UnitCubeMesh(2, 2, 2)
    assert assemble(Constant(m, 1.0) * ds) == approx(6.0)
    m = UnitCubeMesh(1, 1, 1, hexahedral=True)
    assert m.ngs.ne == 1


def test_boundary_ids_match_ds_ids():
    m = UnitSquareMesh(2, 2)
    assert m.boundary_ids("left") == (1,)
    assert m.boundary_ids("left|top") == (1, 4)
    assert m.boundary_ids("bottom|right|top|left") == (1, 2, 3, 4)
    assert m.boundary_ids("nothing") == ()
    one = Constant(m, 1.0)
    for pat in ("bottom", "left|top", "right"):
        ids = m.boundary_ids(pat)
        assert assemble(one * ds(ids)) == approx(assemble(one * ds(pat)))
    b = BoxMesh(1, 1, 1)
    assert b.boundary_ids("top") == (6,)
    assert b.boundary_ids("back|left") == (1, 3)


def test_mesh_materials_default():
    m = UnitSquareMesh(2, 2)
    assert len(m.materials) >= 1
    assert assemble(Constant(m, 1.0) * dx(m.materials[0], domain=m)) == approx(1.0)


def test_mesh_from_occ_disk():
    from netgen.occ import Circle, OCCGeometry
    disk = Circle((0, 0), 1).Face()
    disk.edges.name = "rim"
    m = Mesh(OCCGeometry(disk, dim=2).GenerateMesh(maxh=0.3)).curve(3)
    assert m.dim == 2
    assert m.boundaries == ("rim",)
    assert assemble(Constant(m, 1.0) * dx(domain=m), ) == approx(math.pi, rel=1e-4)
    assert assemble(Constant(m, 1.0) * ds("rim")) == approx(2 * math.pi, rel=1e-4)
    x = SpatialCoordinate(m)
    assert assemble(dot(x, FacetNormal(m)) * ds) == approx(2 * math.pi, rel=1e-4)
    V = FunctionSpace(m, "Lagrange", 2)
    assert V.dim() == m.ngs.nv + m.ngs.nedge


def test_mesh_from_occ_box_3d_multiple_names():
    from netgen.occ import Box, OCCGeometry, Pnt
    b = Box(Pnt(0, 0, 0), Pnt(2, 1, 1))
    from netgen.occ import X
    b.faces.name = "wall"
    b.faces.Min(X).name = "xmin"
    b.faces.Max(X).name = "xmax"
    m = Mesh(OCCGeometry(b).GenerateMesh(maxh=0.5))
    assert m.dim == 3
    assert "xmin" in m.boundaries and "xmax" in m.boundaries
    assert assemble(Constant(m, 1.0) * dx(domain=m)) == approx(2.0)
    assert assemble(Constant(m, 1.0) * ds("xmin")) == approx(1.0)
    assert assemble(Constant(m, 1.0) * ds("xmax")) == approx(1.0)


def test_mesh_from_ngsolve_mesh_object():
    from ngsolve import meshes
    g = meshes.MakeStructured2DMesh(nx=2, ny=2)
    m = Mesh(g)
    assert m.ngs is g
    assert m.dim == 2


def test_mesh_from_netgen_mesh():
    from ngsolve.meshes import MakeStructured2DMesh
    ng = MakeStructured2DMesh(nx=2, ny=2).ngmesh
    m = Mesh(ng)
    assert m.dim == 2
    assert assemble(Constant(m, 1.0) * dx(domain=m)) == approx(1.0)


def test_mesh_from_file(tmp_path):
    g = UnitSquareMesh(2, 2).ngs
    fn = str(tmp_path / "m.vol")
    g.ngmesh.Save(fn)
    m = Mesh(fn)
    assert m.dim == 2 and m.ngs.ne == 8
    assert m.boundaries == ("bottom", "right", "top", "left")


def test_mesh_ufl_properties():
    m = UnitSquareMesh(2, 2)
    assert m.geometric_dimension() == 2
    cn = lambda mm: (lambda c: c() if callable(c) else c)(mm.ufl_cell().cellname)
    assert cn(m) == "triangle"
    assert cn(UnitSquareMesh(2, 2, quadrilateral=True)) == "quadrilateral"
    assert cn(BoxMesh(1, 1, 1)) == "tetrahedron"
    assert cn(BoxMesh(1, 1, 1, hexahedral=True)) == "hexahedron"
    assert cn(UnitIntervalMesh(2)) == "interval"
    assert SpatialCoordinate(m).ufl_shape == (2,)
    assert FacetNormal(m).ufl_shape == (2,)


def test_two_meshes_independent_domains():
    m1, m2 = UnitSquareMesh(2, 2), UnitSquareMesh(4, 4)
    assert m1.ufl_id() != m2.ufl_id()
    assert assemble(Constant(m1, 1.0) * dx(domain=m1)) == approx(1.0)
    assert FunctionSpace(m1, "Lagrange", 1).dim() == 9
    assert FunctionSpace(m2, "Lagrange", 1).dim() == 25


def test_cell_quantity_cache_per_mesh():
    m1, m2 = UnitSquareMesh(2, 2), UnitSquareMesh(4, 4)
    V1, V2 = FunctionSpace(m1, "DG", 0), FunctionSpace(m2, "DG", 0)
    h1 = assemble(CellDiameter(m1) * TestFunction(V1) * dx).FV().NumPy()
    h2 = assemble(CellDiameter(m2) * TestFunction(V2) * dx).FV().NumPy()
    assert h1.sum() == approx(math.sqrt(0.5), rel=1e-12)
    assert h2.sum() == approx(math.sqrt(0.5) / 2, rel=1e-12)


def test_project_into_mixed_space():
    m = mesh("tri")
    W = FunctionSpace(m, "Lagrange", 2) * FunctionSpace(m, "Lagrange", 2)
    w = project((1.0, 2.0), W)
    assert w.sub(1).at(0.3, 0.3) == approx(2.0)


def test_interpolate_into_mixed_space():
    m = mesh("tri")
    W = FunctionSpace(m, "Lagrange", 2) * FunctionSpace(m, "Lagrange", 2)
    w = Function(W).interpolate((1.0, 2.0))
    assert w.sub(1).at(0.3, 0.3) == approx(2.0)


def test_function_vector_accessor():
    m = mesh("tri")
    f = Function(FunctionSpace(m, "Lagrange", 1))
    f.vector()


def test_sub_of_non_mixed_function():
    m = mesh("tri")
    f = Function(FunctionSpace(m, "Lagrange", 1))
    assert f.sub(0) is f and f.subfunctions == (f,)
    g = Function(VectorFunctionSpace(m, "Lagrange", 1))
    assert g.subfunctions == (g,) and g.sub(1).function_space().dim() == f.function_space().dim()

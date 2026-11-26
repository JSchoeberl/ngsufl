"""Firedrake compatibility: scripts written for Firedrake run after changing the import line."""

import math

import matplotlib.pyplot as plt
import ngsolve as ngs
import numpy as np
import pytest

from ngsufl import *
import ngsufl
import ngsufl.pyplot as fplt
from conftest import approx, close, dense, native_bf, native_lf

pi_ = math.pi


def meas(mesh):
    return assemble(Constant(1.0) * dx(domain=mesh))


def coord(mesh, i, *ids):
    x = SpatialCoordinate(mesh)
    return assemble(x[i] * (ds(*ids) if ids else dx(domain=mesh)))


def bnd_len(mesh, i):
    return assemble(Constant(1.0) * ds(i, domain=mesh))


def xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


# ---------------------------------------------------------------- meshes

MESH_SIGS = [
    ("interval_len", lambda **k: IntervalMesh(4, 2.0, **k), 1, 4, 2.0),
    ("interval_lr", lambda **k: IntervalMesh(4, 1.0, 3.0, **k), 1, 4, 2.0),
    ("unit_interval", lambda **k: UnitIntervalMesh(5, **k), 1, 5, 1.0),
    ("unit_square", lambda **k: UnitSquareMesh(3, 2, **k), 2, 12, 1.0),
    ("square", lambda **k: SquareMesh(3, 3, 2.0, **k), 2, 18, 4.0),
    ("rectangle", lambda **k: RectangleMesh(2, 3, 2.0, 3.0, **k), 2, 12, 6.0),
    ("unit_cube", lambda **k: UnitCubeMesh(2, 2, 1, **k), 3, 24, 1.0),
    ("cube", lambda **k: CubeMesh(1, 1, 1, 2.0, **k), 3, 6, 8.0),
    ("box", lambda **k: BoxMesh(2, 1, 1, 2.0, 1.0, 3.0, **k), 3, 12, 6.0),
]

PARALLEL_KW = [
    {}, {"comm": None}, {"reorder": True}, {"reorder": False},
    {"distribution_parameters": {"partition": False}}, {"name": "mymesh"},
]


@pytest.mark.parametrize("sig", MESH_SIGS, ids=[s[0] for s in MESH_SIGS])
def test_mesh_measure_and_cells(sig):
    _, make, dim, ncells, vol = sig
    m = make()
    assert m.geometric_dimension() == dim
    assert m.topological_dimension() == dim
    assert m.num_cells() == ncells
    assert meas(m) == pytest.approx(vol)


@pytest.mark.parametrize("sig", MESH_SIGS, ids=[s[0] for s in MESH_SIGS])
@pytest.mark.parametrize("kw", PARALLEL_KW, ids=lambda k: "-".join(k) or "none")
def test_mesh_parallel_keywords_ignored(sig, kw):
    _, make, dim, ncells, vol = sig
    m = make(**kw)
    assert m.num_cells() == ncells
    assert meas(m) == pytest.approx(vol)


def test_mesh_num_vertices():
    assert UnitIntervalMesh(7).num_vertices() == 8
    assert UnitSquareMesh(3, 4).num_vertices() == 20
    assert UnitSquareMesh(3, 4, quadrilateral=True).num_vertices() == 20
    assert UnitCubeMesh(2, 3, 4).num_vertices() == 60
    assert BoxMesh(2, 3, 4, 1, 1, 1, hexahedral=True).num_vertices() == 60


def test_quadrilateral_and_hexahedral_cell_counts():
    assert RectangleMesh(3, 4, 1, 1, quadrilateral=True).num_cells() == 12
    assert UnitSquareMesh(3, 4, quadrilateral=True).num_cells() == 12
    assert SquareMesh(5, 2, 1.0, quadrilateral=True).num_cells() == 10
    assert BoxMesh(2, 3, 4, 1, 1, 1, hexahedral=True).num_cells() == 24
    assert UnitCubeMesh(2, 2, 2, hexahedral=True).num_cells() == 8
    assert CubeMesh(2, 2, 2, 3.0, hexahedral=True).num_cells() == 8
    assert meas(CubeMesh(2, 2, 2, 3.0, hexahedral=True)) == pytest.approx(27.0)


def test_simplex_cell_counts():
    assert UnitSquareMesh(3, 4).num_cells() == 24
    assert UnitCubeMesh(2, 2, 2).num_cells() == 48
    assert UnitCubeMesh(1, 1, 1).num_cells() == 6


def test_ufl_cells():
    assert UnitSquareMesh(1, 1).ufl_cell().cellname ==  "triangle"
    assert UnitSquareMesh(1, 1, quadrilateral=True).ufl_cell().cellname ==  "quadrilateral"
    assert UnitCubeMesh(1, 1, 1).ufl_cell().cellname ==  "tetrahedron"
    assert UnitCubeMesh(1, 1, 1, hexahedral=True).ufl_cell().cellname ==  "hexahedron"
    assert UnitIntervalMesh(2).ufl_cell().cellname ==  "interval"


def test_interval_extents_and_boundaries():
    m = IntervalMesh(5, 1.0, 3.0)
    x = SpatialCoordinate(m)
    assert assemble(x[0] * dx) == pytest.approx(4.0)
    assert assemble(Constant(1.0) * ds(1, domain=m)) == pytest.approx(1.0)
    assert assemble(x[0] * ds(1)) == pytest.approx(1.0)
    assert assemble(x[0] * ds(2)) == pytest.approx(3.0)
    assert assemble(x[0] * ds((1, 2))) == pytest.approx(4.0)
    assert assemble(x[0] * ds) == pytest.approx(4.0)


def test_interval_dirichlet_ids():
    m = UnitIntervalMesh(8)
    V = FunctionSpace(m, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == Constant(0.0) * v * dx, uh,
          bcs=[DirichletBC(V, 1.0, 1), DirichletBC(V, 3.0, 2)])
    assert uh.at(0.25) == pytest.approx(1.5)
    assert uh.at(0.75) == pytest.approx(2.5)


@pytest.mark.parametrize("quad", [False, True])
@pytest.mark.parametrize("bid,axis,pos", [(1, 0, 0.0), (2, 0, 2.0), (3, 1, 0.0), (4, 1, 3.0)])
def test_rectangle_boundary_ids(quad, bid, axis, pos):
    m = RectangleMesh(4, 3, 2.0, 3.0, quadrilateral=quad)
    length = 3.0 if axis == 0 else 2.0
    assert bnd_len(m, bid) == pytest.approx(length)
    assert coord(m, axis, bid) == pytest.approx(pos * length)
    other = 1 - axis
    assert coord(m, other, bid) == pytest.approx(0.5 * length * length)


@pytest.mark.parametrize("hexa", [False, True])
@pytest.mark.parametrize("bid,axis,pos", [(1, 0, 0.0), (2, 0, 2.0), (3, 1, 0.0), (4, 1, 3.0), (5, 2, 0.0), (6, 2, 4.0)])
def test_box_boundary_ids(hexa, bid, axis, pos):
    m = BoxMesh(2, 2, 2, 2.0, 3.0, 4.0, hexahedral=hexa)
    L = (2.0, 3.0, 4.0)
    area = np.prod(L) / L[axis]
    assert assemble(Constant(1.0) * ds(bid, domain=m)) == pytest.approx(area)
    assert coord(m, axis, bid) == pytest.approx(pos * area)


def test_unit_square_ids_unit():
    m = UnitSquareMesh(3, 3)
    assert [coord(m, 0, i) for i in (1, 2)] == approx([0.0, 1.0])
    assert [coord(m, 1, i) for i in (3, 4)] == approx([0.0, 1.0])


def test_boundary_id_tuple_and_list():
    m = UnitSquareMesh(4, 4)
    assert assemble(Constant(1.0) * ds((1, 2), domain=m)) == pytest.approx(2.0)
    assert assemble(Constant(1.0) * ds((1, 2, 3, 4), domain=m)) == pytest.approx(4.0)
    assert assemble(Constant(1.0) * ds(domain=m)) == pytest.approx(4.0)


def test_boundary_names_extra():
    m = UnitSquareMesh(4, 4)
    assert assemble(Constant(1.0) * ds("left", domain=m)) == pytest.approx(1.0)
    assert assemble(Constant(1.0) * ds("left|top", domain=m)) == pytest.approx(2.0)
    assert coord(m, 0, "right") == pytest.approx(1.0)
    m3 = UnitCubeMesh(2, 2, 2)
    for name in ("back", "front", "left", "right", "bottom", "top"):
        assert assemble(Constant(1.0) * ds(name, domain=m3)) == pytest.approx(1.0)


@pytest.mark.parametrize("make", [lambda: UnitSquareMesh(2, 2), lambda: UnitIntervalMesh(3),
                                  lambda: UnitCubeMesh(1, 1, 1)])
def test_wrong_boundary_id_raises(make):
    m = make()
    with pytest.raises(ValueError):
        assemble(Constant(1.0) * ds(42, domain=m))


def test_wrong_keyword_raises():
    with pytest.raises(TypeError):
        UnitSquareMesh(2, 2, bogus=1)


def test_mesh_coordinates():
    m = RectangleMesh(2, 3, 2.0, 3.0, originX=1.0, originY=-1.0)
    X = m.coordinates
    assert X.function_space().dim() == 2 * m.num_vertices()
    d = X.dat.data_ro
    assert d.shape == (m.num_vertices(), 2)
    assert d.min(axis=0) == approx([1.0, -1.0])
    assert d.max(axis=0) == approx([2.0, 3.0])


def test_rectangle_origin_extent():
    m = RectangleMesh(2, 3, 2.0, 3.0, originX=1.0, originY=-1.0)
    assert meas(m) == pytest.approx(1.0 * 4.0)
    x = SpatialCoordinate(m)
    assert assemble(x[0] * dx) == pytest.approx(1.5 * 4.0)
    assert assemble(x[1] * dx) == pytest.approx(1.0 * 4.0)


def test_rectangle_origin_boundaries():
    m = RectangleMesh(3, 3, 2.0, 3.0, originX=1.0, originY=-1.0)
    assert coord(m, 0, 1) == pytest.approx(1.0 * 4.0)
    assert coord(m, 0, 2) == pytest.approx(2.0 * 4.0)
    assert coord(m, 1, 3) == pytest.approx(-1.0 * 1.0)
    assert coord(m, 1, 4) == pytest.approx(3.0 * 1.0)


def test_coordinates_are_p1_vector():
    m = UnitSquareMesh(3, 3)
    X = m.coordinates
    x = SpatialCoordinate(m)
    assert assemble(inner(X - x, X - x) * dx) == pytest.approx(0.0, abs=1e-20)


def test_mesh_coordinates_3d_shape():
    m = BoxMesh(1, 2, 3, 1.0, 2.0, 3.0)
    d = m.coordinates.dat.data_ro
    assert d.shape == (m.num_vertices(), 3)
    assert d.max(axis=0) == approx([1.0, 2.0, 3.0])


def test_diagonal_options_differ():
    vals = {}
    for d in ("left", "right"):
        m = UnitSquareMesh(1, 1, diagonal=d)
        x = SpatialCoordinate(m)
        vals[d] = assemble(avg(x[0] * x[1]) * dS)
    assert m.num_cells() == 2
    assert sorted(vals.values()) == approx(sorted([math.sqrt(2) / 3, math.sqrt(2) / 6]))


@pytest.mark.parametrize("d", ["left", "right"])
def test_diagonal_rectangle_and_square(d):
    assert RectangleMesh(2, 2, 1, 1, diagonal=d).num_cells() == 8
    assert SquareMesh(2, 2, 1.0, diagonal=d).num_cells() == 8
    assert meas(UnitSquareMesh(3, 3, diagonal=d)) == pytest.approx(1.0)


def test_quadrilateral_with_diagonal_raises():
    with pytest.raises(ValueError):
        UnitSquareMesh(2, 2, quadrilateral=True, diagonal="left")


@xfail("diagonal='crossed' is not implemented")
def test_diagonal_crossed():
    m = UnitSquareMesh(2, 2, diagonal="crossed")
    assert m.num_cells() == 16


def test_functionspace_mesh_accessor():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "CG", 1)
    assert V.mesh() is m
    assert Function(V).function_space().mesh() is m


# ---------------------------------------------------------------- spaces

M2 = UnitSquareMesh(2, 2)
M3 = UnitCubeMesh(1, 1, 1)
M1 = IntervalMesh(4, 1.0)
MQ = UnitSquareMesh(2, 2, quadrilateral=True)
MH = UnitCubeMesh(1, 1, 1, hexahedral=True)

DIMS = [
    (M1, "CG", 1, 5), (M1, "CG", 2, 9), (M1, "DG", 0, 4), (M1, "DG", 2, 12),
    (M2, "CG", 1, 9), (M2, "CG", 2, 25), (M2, "CG", 3, 49), (M2, "Lagrange", 2, 25),
    (M2, "P", 2, 25), (M2, "DG", 0, 8), (M2, "DG", 1, 24), (M2, "DG", 2, 48),
    (M2, "RT", 1, 16), (M2, "RT", 2, 48), (M2, "BDM", 1, 32), (M2, "N1curl", 1, 16),
    (M2, "N2curl", 1, 32), (M2, "CR", 1, 16), (M2, "R", 0, 1), (M2, "Regge", 0, 16),
    (M2, "HHJ", 0, 16),
    (MQ, "Q", 1, 9), (MQ, "Q", 2, 25), (MQ, "DQ", 1, 16), (MQ, "DQ", 0, 4), (MQ, "CG", 1, 9),
    (M3, "CG", 1, 8), (M3, "CG", 2, 27), (M3, "DG", 0, 6), (M3, "DG", 1, 24),
    (M3, "N1curl", 1, 19), (M3, "RT", 1, 18), (M3, "CR", 1, 18),
    (MH, "Q", 1, 8), (MH, "Q", 2, 27), (MH, "DQ", 1, 8),
]


@pytest.mark.parametrize("mesh,family,degree,n", DIMS,
                         ids=[f"{d[0].ufl_cell()}-{d[1]}{d[2]}" for d in DIMS])
def test_function_space_dim(mesh, family, degree, n):
    V = FunctionSpace(mesh, family, degree)
    assert V.dim() == n
    assert Function(V).dat.data.shape[0] == n


def test_vector_and_tensor_space_dims():
    assert VectorFunctionSpace(M2, "CG", 2).dim() == 50
    assert VectorFunctionSpace(M2, "DG", 1).dim() == 48
    assert VectorFunctionSpace(M3, "CG", 1).dim() == 24
    assert VectorFunctionSpace(M1, "CG", 2).dim() == 9
    assert VectorFunctionSpace(M2, "CG", 2, dim=3).dim() == 75
    assert TensorFunctionSpace(M2, "CG", 1).dim() == 36
    assert TensorFunctionSpace(M3, "CG", 1).dim() == 72
    assert TensorFunctionSpace(M2, "DG", 0).dim() == 32


def test_value_shapes():
    assert Function(FunctionSpace(M2, "CG", 1)).ufl_shape == ()
    assert Function(VectorFunctionSpace(M2, "CG", 1)).ufl_shape == (2,)
    assert Function(VectorFunctionSpace(M3, "CG", 1)).ufl_shape == (3,)
    assert Function(TensorFunctionSpace(M2, "CG", 1)).ufl_shape == (2, 2)
    assert Function(FunctionSpace(M2, "RT", 1)).ufl_shape == (2,)
    assert Function(FunctionSpace(M2, "N1curl", 1)).ufl_shape == (2,)
    assert Function(FunctionSpace(M3, "N1curl", 1)).ufl_shape == (3,)
    assert Function(FunctionSpace(M2, "Regge", 0)).ufl_shape == (2, 2)


def test_value_size():
    assert FunctionSpace(M2, "CG", 1).value_size == 1
    assert VectorFunctionSpace(M2, "CG", 1).value_size == 2
    assert TensorFunctionSpace(M3, "CG", 1).value_size == 9


def test_mixed_spaces_dims():
    V = VectorFunctionSpace(M2, "CG", 2)
    Q = FunctionSpace(M2, "CG", 1)
    Z = V * Q
    assert Z.dim() == 59
    assert MixedFunctionSpace([V, Q]).dim() == 59
    assert len(Z) == 2
    assert len(Z.subspaces) == 2
    assert Z.sub(0).collapse().dim() == 50
    assert Z.sub(1).collapse().dim() == 9
    W = FunctionSpace(M2, "BDM", 1) * FunctionSpace(M2, "DG", 0)
    assert W.dim() == 40
    W3 = FunctionSpace(M2, "RT", 1) * FunctionSpace(M2, "DG", 0) * FunctionSpace(M2, "CG", 1)
    assert W3.dim() == 16 + 8 + 9
    assert len(W3) == 3


def test_mixed_function_space_sub_value_shapes():
    Z = VectorFunctionSpace(M2, "CG", 2) * FunctionSpace(M2, "CG", 1)
    w = Function(Z)
    u, p = w.subfunctions
    assert u.ufl_shape == (2,)
    assert p.ufl_shape == ()
    assert u.function_space().dim() == 50


def test_split_of_mixed_function_in_forms():
    Z = VectorFunctionSpace(M2, "CG", 2) * FunctionSpace(M2, "CG", 1)
    w = Function(Z)
    u, p = split(w)
    assert u.ufl_shape == (2,)
    assert p.ufl_shape == ()
    u, p = TrialFunctions(Z)
    v, q = TestFunctions(Z)
    assert u.ufl_shape == (2,) and q.ufl_shape == ()


def test_element_objects():
    V = FunctionSpace(M2, FiniteElement("CG", triangle, 2))
    assert V.dim() == 25
    W = FunctionSpace(M2, VectorElement("CG", triangle, 2))
    assert W.dim() == 50
    T = FunctionSpace(M2, TensorElement("CG", triangle, 1))
    assert T.dim() == 36
    Z = FunctionSpace(M2, MixedElement([VectorElement("CG", triangle, 2), FiniteElement("CG", triangle, 1)]))
    assert Z.dim() == 59
    E = VectorElement("CG", triangle, 2) * FiniteElement("CG", triangle, 1)
    assert FunctionSpace(M2, E).dim() == 59


def test_element_alias_families():
    for fam in ("CG", "Lagrange", "P"):
        assert FunctionSpace(M2, fam, 2).dim() == 25
    for fam in ("DG", "Discontinuous Lagrange"):
        assert FunctionSpace(M2, fam, 1).dim() == 24
    assert FunctionSpace(M2, "Raviart-Thomas", 1).dim() == 16
    assert FunctionSpace(M2, "Brezzi-Douglas-Marini", 1).dim() == 32
    assert FunctionSpace(M2, "Crouzeix-Raviart", 1).dim() == 16


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        FunctionSpace(M2, "NoSuchFamily", 1)


def test_space_mesh_and_cell_of_quad_space():
    V = FunctionSpace(MQ, "Q", 2)
    assert V.mesh() is MQ
    assert V.ufl_element().degree == 2


# ---------------------------------------------------------------- Function API

@pytest.fixture(scope="module")
def sq():
    return UnitSquareMesh(4, 4)


def test_function_name_and_rename(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V, name="Temperature")
    assert u.name() == "Temperature"
    u.rename("T2")
    assert u.name() == "T2"
    assert Function(V, name="a").function_space() is V


def test_function_space_call(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V)
    assert u.function_space() is V
    assert u.function_space().dim() == 25


def test_interpolate_returns_function_and_values(sq):
    V = FunctionSpace(sq, "CG", 1)
    x, y = SpatialCoordinate(sq)
    u = Function(V)
    assert u.interpolate(1 + 2 * x - y) is u
    for p in [(0.3, 0.4), (0.7, 0.2), (0.12, 0.93)]:
        assert u.at(p) == pytest.approx(1 + 2 * p[0] - p[1])


@pytest.mark.parametrize("family,degree", [("CG", 1), ("CG", 2), ("CG", 3), ("DG", 1), ("DG", 2)])
def test_interpolate_polynomial_exact(sq, family, degree):
    V = FunctionSpace(sq, family, degree)
    x, y = SpatialCoordinate(sq)
    f = x * y + 0.5 * x if degree >= 2 else 1 + x - 2 * y
    u = Function(V).interpolate(f)
    for p in [(0.3, 0.4), (0.7, 0.2), (0.12, 0.93)]:
        exact = p[0] * p[1] + 0.5 * p[0] if degree >= 2 else 1 + p[0] - 2 * p[1]
        assert u.at(p) == pytest.approx(exact)


def test_interpolate_vector_and_tensor(sq):
    V = VectorFunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    u = Function(V).interpolate(as_vector((x * y, 1 + x)))
    assert u.at((0.3, 0.4)) == approx([0.12, 1.3])
    T = TensorFunctionSpace(sq, "CG", 1)
    t = Function(T).interpolate(as_matrix(((x, 1), (2, y))))
    assert np.asarray(t.at((0.3, 0.4))).ravel() == approx([0.3, 1.0, 2.0, 0.4])


def test_interpolate_constant_and_numbers(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(Constant(3.0))
    assert assemble(u * dx) == pytest.approx(3.0)
    u.interpolate(2.0)
    assert assemble(u * dx) == pytest.approx(2.0)


def test_interpolate_function_between_spaces(sq):
    V1 = FunctionSpace(sq, "CG", 1)
    V2 = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    u = Function(V1).interpolate(x + y)
    w = Function(V2).interpolate(u)
    assert w.at((0.31, 0.62)) == pytest.approx(0.93)


def test_assign_function_and_numbers(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(SpatialCoordinate(sq)[0])
    w = Function(V)
    assert w.assign(u) is w
    assert w.at((0.3, 0.3)) == pytest.approx(0.3)
    w.assign(2)
    assert assemble(w * dx) == pytest.approx(2.0)
    w.assign(0)
    assert assemble(w * dx) == 0.0
    w.assign(1.5)
    assert assemble(w * dx) == pytest.approx(1.5)


def test_assign_linear_combination(sq):
    V = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    a = Function(V).interpolate(x)
    b = Function(V).interpolate(y * y)
    w = Function(V)
    w.assign(2 * a - 3 * b + 1)
    for p in [(0.3, 0.4), (0.7, 0.2)]:
        assert w.at(p) == pytest.approx(2 * p[0] - 3 * p[1] ** 2 + 1)
    w.assign(a + b)
    assert w.at((0.3, 0.4)) == pytest.approx(0.3 + 0.16)
    w.assign(0.5 * (a + b))
    assert w.at((0.3, 0.4)) == pytest.approx(0.5 * 0.46)


def test_assign_nonlinear_expression(sq):
    V = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    a = Function(V).interpolate(x)
    w = Function(V).assign(a * a)
    assert w.at((0.5, 0.5)) == pytest.approx(0.25)


@pytest.mark.parametrize("family,degree", [("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1)])
def test_assign_self_reference(sq, family, degree):
    V = FunctionSpace(sq, family, degree)
    u = Function(V).assign(1.0)
    u.assign(u + 1)
    assert assemble(u * dx) == pytest.approx(2.0)
    u.assign(2 * u)
    assert assemble(u * dx) == pytest.approx(4.0)
    u.interpolate(u / 2 + 1)
    assert assemble(u * dx) == pytest.approx(3.0)


def test_assign_via_temporary_is_safe(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(1.0)
    old = u.copy(deepcopy=True)
    u.assign(old + 1)
    assert assemble(u * dx) == pytest.approx(2.0)


def test_function_inplace_operators(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(1.0)
    alias = u
    u += 2
    u *= 2
    assert u is alias and isinstance(u, Function)
    assert assemble(alias * dx) == pytest.approx(6.0)


def test_zero_and_copy(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(2.0)
    c = u.copy(deepcopy=True)
    u.zero()
    assert assemble(u * dx) == 0.0
    assert assemble(c * dx) == pytest.approx(2.0)


def test_zero_returns_function(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(1.0)
    assert u.zero() is u


def test_function_val_argument_is_function(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(3.0)
    c = Function(V, val=u)
    assert assemble(c * dx) == pytest.approx(3.0)


@xfail("Function(f) with a Function as first argument (uses its space) raises ValueError")
def test_function_from_function_space_of_function(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(3.0)
    c = Function(u)
    assert c.function_space() is V


def test_project_method_and_function(sq):
    V = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    f = x * x + y
    u = Function(V).project(f)
    assert u.at((0.3, 0.4)) == pytest.approx(0.09 + 0.4)
    w = project(f, V)
    assert w.at((0.3, 0.4)) == pytest.approx(0.49)
    assert assemble((u - w) ** 2 * dx) == pytest.approx(0.0, abs=1e-18)


def test_project_to_lower_space(sq):
    V = FunctionSpace(sq, "CG", 1)
    x, y = SpatialCoordinate(sq)
    w = project(x * x, V)
    ref = Function(V)
    v, u = TestFunction(V), TrialFunction(V)
    solve(u * v * dx == x * x * v * dx, ref)
    assert assemble((w - ref) ** 2 * dx) == pytest.approx(0.0, abs=1e-20)


def test_project_vector_from_function(sq):
    V = VectorFunctionSpace(sq, "CG", 2)
    W = VectorFunctionSpace(sq, "DG", 2)
    x, y = SpatialCoordinate(sq)
    u = Function(V).interpolate(as_vector((x * y, y)))
    w = project(u, W)
    assert w.at((0.3, 0.4)) == approx([0.12, 0.4])


def test_project_with_name(sq):
    V = FunctionSpace(sq, "CG", 1)
    assert project(Constant(1.0), V, name="proj").name() == "proj"


def test_interpolate_function_namespace(sq):
    V = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    u = interpolate(x * y, V)
    assert isinstance(u, Function)
    assert u.at((0.5, 0.6)) == pytest.approx(0.3)


def test_assemble_of_interpolate(sq):
    V = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    u = assemble(interpolate(x * y, V))
    assert isinstance(u, Function)
    assert u.at((0.5, 0.6)) == pytest.approx(0.3)


def test_at_single_point_forms(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(SpatialCoordinate(sq)[0] + 2 * SpatialCoordinate(sq)[1])
    ref = 0.3 + 0.8
    assert u.at((0.3, 0.4)) == pytest.approx(ref)
    assert u.at([0.3, 0.4]) == pytest.approx(ref)
    assert u.at(0.3, 0.4) == pytest.approx(ref)
    assert u((0.3, 0.4)) == pytest.approx(ref)


def test_call_with_numpy_point(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(SpatialCoordinate(sq)[0])
    assert u(np.array([0.3, 0.4])) == pytest.approx(0.3)


def test_at_many_points(sq):
    V = FunctionSpace(sq, "CG", 1)
    x, y = SpatialCoordinate(sq)
    u = Function(V).interpolate(x + 2 * y)
    pts = [(0.3, 0.4), (0.7, 0.2), (0.15, 0.9)]
    got = u.at(pts)
    assert np.asarray(got) == approx([0.3 + 0.8, 0.7 + 0.4, 0.15 + 1.8])
    assert np.asarray(u.at(np.array(pts))) == approx(got)


@xfail("at() of a tensor valued function returns a flat tuple, Firedrake returns an array of the value shape")
def test_at_tensor_function_shape(sq):
    T = TensorFunctionSpace(sq, "CG", 1)
    x, y = SpatialCoordinate(sq)
    t = Function(T).interpolate(as_matrix(((x, 1), (2, y))))
    assert np.asarray(t.at((0.3, 0.4))).shape == (2, 2)


def test_at_vector_function(sq):
    V = VectorFunctionSpace(sq, "CG", 1)
    x, y = SpatialCoordinate(sq)
    u = Function(V).interpolate(as_vector((x, 2 * y)))
    assert u.at((0.3, 0.4)) == approx([0.3, 0.8])
    got = u.at([(0.3, 0.4), (0.5, 0.5)])
    assert np.asarray(got).shape == (2, 2)
    assert np.asarray(got) == approx([[0.3, 0.8], [0.5, 1.0]])


def test_at_boundary_and_vertex_points(sq):
    V = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    u = Function(V).interpolate(x * y)
    assert u.at((1.0, 1.0)) == pytest.approx(1.0)
    assert u.at((0.0, 0.5)) == pytest.approx(0.0)
    assert u.at((0.5, 1.0)) == pytest.approx(0.5)


def test_at_on_1d_mesh():
    m = UnitIntervalMesh(8)
    V = FunctionSpace(m, "CG", 2)
    u = Function(V).interpolate(SpatialCoordinate(m)[0] ** 2)
    assert u.at(0.3) == pytest.approx(0.09)
    assert u.at((0.3,)) == pytest.approx(0.09)


def test_at_on_quad_and_3d():
    mq = UnitSquareMesh(3, 3, quadrilateral=True)
    xq, yq = SpatialCoordinate(mq)
    uq = Function(FunctionSpace(mq, "CG", 2)).interpolate(xq * yq)
    assert uq.at((0.3, 0.4)) == pytest.approx(0.12)
    m3 = UnitCubeMesh(2, 2, 2)
    x, y, z = SpatialCoordinate(m3)
    u3 = Function(FunctionSpace(m3, "CG", 1)).interpolate(x + y + z)
    assert u3.at((0.3, 0.4, 0.2)) == pytest.approx(0.9)


@xfail("at() raises NgException for points outside the mesh, Firedrake raises PointNotInDomainError / returns None with dont_raise=True")
def test_at_dont_raise(sq):
    u = Function(FunctionSpace(sq, "CG", 1))
    assert u.at((5.0, 5.0), dont_raise=True) is None


@xfail("at() outside the mesh raises ngsolve's NgException instead of a Python/Firedrake error type")
def test_at_outside_raises_nice_error(sq):
    u = Function(FunctionSpace(sq, "CG", 1))
    with pytest.raises((ValueError, RuntimeError, LookupError)):
        u.at((5.0, 5.0))


def test_dat_data_p1_scalar(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V)
    d = u.dat.data
    assert d.shape == (25,)
    d[:] = 3.0
    assert assemble(u * dx) == pytest.approx(3.0)
    assert u.dat.data_ro.shape == (25,)


def test_dat_data_write_through_slices(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(1.0)
    u.dat.data[:] = np.arange(25.0)
    assert u.vector().get_local() == approx(np.arange(25.0))
    u.dat.data[0] = 100.0
    assert u.vector().get_local()[0] == 100.0
    u.dat.data[:] += 1.0
    assert u.vector().get_local()[1] == pytest.approx(2.0)


def test_dat_data_ro_values_p1(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(SpatialCoordinate(sq)[0])
    assert sorted(set(np.round(u.dat.data_ro, 12))) == approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_dat_data_dg0_values():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "DG", 0)
    u = Function(V).assign(3.0)
    assert u.dat.data.shape == (8,)
    assert u.dat.data == approx(np.full(8, 3.0))


def test_dat_data_vector_shape_and_write(sq):
    V = VectorFunctionSpace(sq, "CG", 1)
    u = Function(V)
    assert u.dat.data.shape == (25, 2)
    u.dat.data[:, 0] = 1.0
    u.dat.data[:, 1] = 2.0
    assert u.at((0.3, 0.4)) == approx([1.0, 2.0])
    u.dat.data[:] = (3.0, 4.0)
    assert u.at((0.3, 0.4)) == approx([3.0, 4.0])


def test_dat_data_vector_dg_shape():
    m = UnitSquareMesh(2, 2)
    assert Function(VectorFunctionSpace(m, "DG", 1)).dat.data.shape == (24, 2)
    assert Function(VectorFunctionSpace(m, "CG", 2)).dat.data.shape == (25, 2)
    assert Function(TensorFunctionSpace(m, "CG", 1)).dat.data.shape == (9, 2, 2)


def test_dat_data_mixed_tuple():
    Z = VectorFunctionSpace(M2, "CG", 2) * FunctionSpace(M2, "CG", 1)
    w = Function(Z)
    d = w.dat.data
    assert isinstance(d, tuple) and len(d) == 2
    assert d[0].shape == (25, 2)
    assert d[1].shape == (9,)
    d[1][:] = 4.0
    assert w.subfunctions[1].at((0.3, 0.3)) == pytest.approx(4.0)


def test_dat_data_with_halos_aliases(sq):
    u = Function(FunctionSpace(sq, "CG", 1))
    assert u.dat.data_ro.shape == u.dat.data_with_halos.shape == u.dat.data_ro_with_halos.shape


def test_vector_interface(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(SpatialCoordinate(sq)[0])
    vec = u.vector()
    a = vec.get_local()
    assert a.shape == (25,)
    assert vec.size() == 25 if callable(vec.size) else vec.size == 25
    assert vec.array() == approx(a)
    assert vec.max() == pytest.approx(1.0)
    assert vec.min() == pytest.approx(0.0)
    assert vec.norm() == pytest.approx(np.linalg.norm(a))
    vec.set_local(2 * a)
    assert u.at((1.0, 0.5)) == pytest.approx(2.0)
    assert vec[:].shape == (25,)


def test_vector_of_function_shares_data(sq):
    u = Function(FunctionSpace(sq, "CG", 1)).assign(1.0)
    u.vector().set_local(np.full(25, 5.0))
    assert assemble(u * dx) == pytest.approx(5.0)
    u.vector().zero()
    assert assemble(u * dx) == 0.0


def test_gf_attribute_is_ngsolve(sq):
    V = FunctionSpace(sq, "CG", 2)
    u = Function(V).interpolate(SpatialCoordinate(sq)[0])
    assert isinstance(u.gf, ngs.GridFunction)
    assert ngs.Integrate(u.gf, sq.ngs) == pytest.approx(0.5)


def test_mixed_function_sub_and_subfunctions():
    Z = VectorFunctionSpace(M2, "CG", 2) * FunctionSpace(M2, "CG", 1)
    w = Function(Z)
    u, p = w.subfunctions
    assert w.sub(0) is u or w.sub(0).function_space().dim() == 50
    p.assign(3.0)
    assert assemble(w.sub(1) * dx(domain=M2)) == pytest.approx(3.0)
    assert assemble(split(w)[1] * dx) == pytest.approx(3.0)
    u.assign(as_vector((1.0, 2.0)))
    assert u.at((0.3, 0.3)) == approx([1.0, 2.0])


def test_mixed_function_interpolate():
    Z = VectorFunctionSpace(M2, "CG", 2) * FunctionSpace(M2, "CG", 1)
    x, y = SpatialCoordinate(M2)
    w = Function(Z)
    w.sub(0).interpolate(as_vector((x, y)))
    w.sub(1).interpolate(x + y)
    u, p = w.subfunctions
    assert u.at((0.25, 0.5)) == approx([0.25, 0.5])
    assert p.at((0.25, 0.5)) == pytest.approx(0.75)


def test_mixed_function_assemble_through_split():
    Z = FunctionSpace(M2, "CG", 1) * FunctionSpace(M2, "DG", 0)
    w = Function(Z)
    a, b = split(w)
    w.sub(0).assign(2.0)
    w.sub(1).assign(5.0)
    assert assemble((a + b) * dx) == pytest.approx(7.0)


def test_function_in_form_updates_when_changed(sq):
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).assign(1.0)
    form = u * dx
    assert assemble(form) == pytest.approx(1.0)
    u.assign(4.0)
    assert assemble(form) == pytest.approx(4.0)


def test_function_value_shape_in_expressions():
    sq = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(as_vector((1.0, 2.0)))
    assert assemble(inner(u, u) * dx) == pytest.approx(5.0)
    assert assemble(u[1] * dx) == pytest.approx(2.0)
    assert assemble(dot(u, Constant((3.0, 1.0))) * dx) == pytest.approx(5.0)


def test_function_riesz_style_name_default(sq):
    u = Function(FunctionSpace(sq, "CG", 1))
    assert isinstance(u.name(), str)


# ---------------------------------------------------------------- Constant

def test_constant_scalar_vector_tensor(sq):
    c = Constant(2.5)
    assert float(c) == pytest.approx(2.5)
    assert assemble(c * dx(domain=sq)) == pytest.approx(2.5)
    v = Constant((1.0, 2.0))
    assert assemble(v[1] * dx(domain=sq)) == pytest.approx(2.0)
    t = Constant(((1.0, 2.0), (3.0, 4.0)))
    assert assemble(t[1, 0] * dx(domain=sq)) == pytest.approx(3.0)


def test_constant_assign_changes_forms(sq):
    c = Constant(1.0)
    form = c * dx(domain=sq)
    assert assemble(form) == pytest.approx(1.0)
    c.assign(3.0)
    assert assemble(form) == pytest.approx(3.0)
    v = Constant((1.0, 1.0))
    f2 = v[0] * v[1] * dx(domain=sq)
    v.assign((2.0, 3.0))
    assert assemble(f2) == pytest.approx(6.0)


def test_constant_assign_returns_constant():
    c = Constant(1.0)
    assert c.assign(2.0) is c


def test_constant_values_method():
    c = Constant((1.0, 2.0))
    assert np.asarray(c.values()) == approx([1.0, 2.0])


def test_constant_arithmetic_in_expression(sq):
    c = Constant(2.0)
    V = FunctionSpace(sq, "CG", 1)
    u = Function(V).interpolate(c * SpatialCoordinate(sq)[0] + c ** 2)
    assert u.at((0.5, 0.5)) == pytest.approx(5.0)


def test_constant_independent_of_later_meshes():
    m1 = UnitSquareMesh(2, 2)
    c = Constant(2.0)
    m2 = UnitSquareMesh(3, 3)
    u = Function(FunctionSpace(m2, "CG", 1)).assign(1.0)
    assert assemble(c * u * dx) == pytest.approx(2.0)


def test_constant_before_any_mesh(monkeypatch):
    monkeypatch.setattr(ngsufl.mesh, "_last_mesh", None)
    c = Constant(2.0)
    m = UnitSquareMesh(2, 2)
    assert assemble(c * dx(domain=m)) == pytest.approx(2.0)


def test_constant_int_value():
    c = Constant(3)
    assert float(c) == 3.0


def test_constant_with_mesh_first_argument(sq):
    c = Constant(sq, 4.0)
    assert assemble(c * dx(domain=sq)) == pytest.approx(4.0)


# ---------------------------------------------------------------- assemble

def test_assemble_scalar_returns_float(sq):
    r = assemble(Constant(2.0) * dx(domain=sq))
    assert isinstance(r, (float, np.floating))
    assert r == pytest.approx(2.0)


def test_assemble_functional_values(sq):
    x, y = SpatialCoordinate(sq)
    assert assemble(x * y * dx) == pytest.approx(0.25)
    assert assemble(x * x * dx) == pytest.approx(1 / 3)
    assert assemble(sin(pi * x) * dx) == pytest.approx(2 / pi)
    assert assemble(x * ds) == pytest.approx(2.0)
    assert assemble(x * y * dx(degree=2)) == pytest.approx(0.25)


def test_assemble_functional_of_functions(sq):
    V = FunctionSpace(sq, "CG", 2)
    x, y = SpatialCoordinate(sq)
    u = Function(V).interpolate(x * x)
    assert assemble(u * dx) == pytest.approx(1 / 3)
    assert assemble(u * u * dx) == pytest.approx(0.2)
    assert assemble(inner(grad(u), grad(u)) * dx) == pytest.approx(4 / 3)


def test_assemble_1form_is_cofunction(sq):
    V = FunctionSpace(sq, "CG", 1)
    b = assemble(TestFunction(V) * dx)
    assert isinstance(b, Cofunction)
    assert b.dat.data.shape == (25,)
    assert b.dat.data.sum() == pytest.approx(1.0)
    assert b.dat.data_ro.shape == (25,)


def test_assemble_1form_matches_native(sq):
    V = FunctionSpace(sq, "CG", 2)
    f = Function(V).interpolate(SpatialCoordinate(sq)[0])
    b = assemble(f * TestFunction(V) * dx)
    ref = native_lf(V.fes, lambda w: f.gf * w * ngs.dx)
    close(b.vec.FV().NumPy(), ref, 1e-12)


def test_assemble_2form_is_matrix(sq):
    V = FunctionSpace(sq, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(inner(grad(u), grad(v)) * dx)
    assert isinstance(A, Matrix)
    assert A.mat.height == 25
    ref = native_bf(V.fes, lambda U, W: ngs.grad(U) * ngs.grad(W) * ngs.dx)
    close(dense(A.mat), ref, 1e-12)


def test_assemble_mass_matrix_values(sq):
    V = FunctionSpace(sq, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(u * v * dx)
    ref = native_bf(V.fes, lambda U, W: U * W * ngs.dx)
    close(dense(A.mat), ref, 1e-12)
    assert A.array().shape == (81, 81)


def test_assemble_vector_space_matrix_native():
    m = UnitSquareMesh(3, 3)
    V = VectorFunctionSpace(m, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(inner(sym(grad(u)), sym(grad(v))) * dx)
    ref = native_bf(V.fes, lambda U, W: ngs.InnerProduct(ngs.Sym(ngs.Grad(U)), ngs.Sym(ngs.Grad(W))) * ngs.dx)
    close(dense(A.mat), ref, 1e-11)


def test_assemble_matrix_array_with_bcs(sq):
    V = FunctionSpace(sq, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    bc = DirichletBC(V, 0.0, 1)
    A = assemble(inner(grad(u), grad(v)) * dx, bcs=[bc])
    dofs = [i for i, b in enumerate(bc.dofs()) if b]
    assert len(dofs) == 5
    arr = A.array()
    for i in dofs:
        row = arr[i].copy()
        assert row[i] == 1.0
        row[i] = 0.0
        assert np.abs(row).max() == 0.0


def test_assemble_matrix_bcs_single_object(sq):
    V = FunctionSpace(sq, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(u * v * dx, bcs=DirichletBC(V, 0.0, 1))
    assert len(A.bcs) == 1


def test_assemble_matrix_petsc_handle_unsupported(sq):
    V = FunctionSpace(sq, "CG", 1)
    A = assemble(TrialFunction(V) * TestFunction(V) * dx)
    with pytest.raises(NotImplementedError):
        A.M.handle
    with pytest.raises(NotImplementedError):
        A.petscmat


def test_assemble_matrix_vector_product(sq):
    V = FunctionSpace(sq, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(u * v * dx)
    one = Function(V).assign(1.0)
    y = A * one.vector()
    assert y.get_local().sum() == pytest.approx(1.0)


def test_assemble_1form_with_bcs_sets_boundary_values(sq):
    V = FunctionSpace(sq, "CG", 1)
    bc = DirichletBC(V, 7.0, 1)
    b = assemble(TestFunction(V) * dx, bcs=[bc])
    idx = [i for i, f in enumerate(bc.dofs()) if f]
    assert b.dat.data_ro[idx] == approx(np.full(len(idx), 7.0))


def test_assemble_1form_bcs_via_apply(sq):
    V = FunctionSpace(sq, "CG", 1)
    bc = DirichletBC(V, 7.0, 1)
    b = assemble(TestFunction(V) * dx)
    bc.apply(b)
    idx = [i for i, f in enumerate(bc.dofs()) if f]
    assert b.dat.data_ro[idx] == approx(np.full(len(idx), 7.0))


def test_assemble_expression_without_form_passthrough():
    assert assemble(3.0) == 3.0


def test_interior_facet_total_length():
    m = UnitSquareMesh(2, 2)
    inner_len = 1.0 + 1.0 + 4 * math.sqrt(2) / 2
    assert assemble(Constant(1.0) * dS(domain=m)) == pytest.approx(inner_len)


def test_facet_normal_integrals():
    m = UnitSquareMesh(3, 3)
    n = FacetNormal(m)
    assert assemble(n[0] * ds(2)) == pytest.approx(1.0)
    assert assemble(n[0] * ds(1)) == pytest.approx(-1.0)
    assert assemble(n[1] * ds(4)) == pytest.approx(1.0)
    assert assemble(n[1] * ds(3)) == pytest.approx(-1.0)
    assert assemble(inner(Constant((1.0, 2.0)), n) * ds) == pytest.approx(0.0, abs=1e-14)


def test_facet_normal_3d():
    m = UnitCubeMesh(2, 2, 2)
    n = FacetNormal(m)
    assert assemble(n[2] * ds(6)) == pytest.approx(1.0)
    assert assemble(n[2] * ds(5)) == pytest.approx(-1.0)
    assert assemble(n[0] * ds(1)) == pytest.approx(-1.0)


def test_cell_size_and_volume():
    m = UnitSquareMesh(4, 4)
    h = CellSize(m)
    d = Function(FunctionSpace(m, "DG", 0)).interpolate(h)
    assert d.dat.data.max() == pytest.approx(math.sqrt(2) / 4)
    assert assemble(CellVolume(m) * dx) == pytest.approx(1.0 / 32)
    hd = Function(FunctionSpace(m, "DG", 0)).interpolate(CellDiameter(m))
    assert hd.dat.data.min() == pytest.approx(math.sqrt(2) / 4)


def test_spatial_coordinate_expressions():
    m = RectangleMesh(4, 4, 2.0, 1.0)
    x, y = SpatialCoordinate(m)
    assert assemble(x * dx) == pytest.approx(2.0)
    assert assemble(y * dx) == pytest.approx(1.0)
    assert assemble(x * y * dx) == pytest.approx(1.0)
    assert assemble(sqrt(x * x + y * y) * dx) > 0.0


def test_ufl_namespace_functions(sq):
    x, y = SpatialCoordinate(sq)
    assert assemble(exp(x) * dx) == pytest.approx(math.e - 1)
    assert assemble(conditional(lt(x, 0.5), 1.0, 0.0) * dx) == pytest.approx(0.5)
    assert assemble(max_value(x, 0.5) * dx) == pytest.approx(0.625)
    assert assemble(abs(x - 0.5) * dx) == pytest.approx(0.25)
    assert assemble(pi * dx(domain=sq)) == pytest.approx(math.pi)
    assert assemble(cos(pi * x) ** 2 * dx) == pytest.approx(0.5)
    assert assemble(sign(x - 0.25) * dx) == pytest.approx(0.5)


def test_measures_with_subdomain_degree_metadata(sq):
    x, y = SpatialCoordinate(sq)
    assert assemble(x**4 * dx(degree=6)) == pytest.approx(0.2)
    assert assemble(x * dx(metadata={"quadrature_degree": 2})) == pytest.approx(0.5)


# ---------------------------------------------------------------- DirichletBC

def test_dirichlet_bc_number_on_ids():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V, 2.0, 1).apply(u)
    assert u.at((0.0, 0.5)) == pytest.approx(2.0)
    assert u.at((0.5, 0.5)) == 0.0
    assert u.at((1.0, 0.5)) == 0.0


@pytest.mark.parametrize("sub,count", [(1, 5), ((1, 2), 10), ([1, 2], 10), ((1, 2, 3, 4), 16),
                                       ("on_boundary", 16)])
def test_dirichlet_bc_sub_domain_variants(sub, count):
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V, 1.0, sub).apply(u)
    assert int(round(u.dat.data.sum())) == count


@pytest.mark.parametrize("make_g", [
    lambda m: 2.5,
    lambda m: Constant(2.5),
    lambda m: Function(FunctionSpace(m, "CG", 1)).assign(2.5),
    lambda m: 2.0 + Constant(0.5),
    lambda m: 0 * SpatialCoordinate(m)[0] + 2.5,
], ids=["number", "Constant", "Function", "expression", "ufl_x"])
def test_dirichlet_bc_value_kinds(make_g):
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V, make_g(m), 3).apply(u)
    assert u.at((0.5, 0.0)) == pytest.approx(2.5)
    assert u.at((0.5, 0.25)) == 0.0


def test_dirichlet_bc_ufl_expression_of_x():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 2)
    x, y = SpatialCoordinate(m)
    u = Function(V)
    DirichletBC(V, x * x + 1, (3, 4)).apply(u)
    assert u.at((0.5, 0.0)) == pytest.approx(1.25)
    assert u.at((0.3, 1.0)) == pytest.approx(1.09)


def test_dirichlet_bc_vector_values():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V, Constant((1.0, 2.0)), 4).apply(u)
    assert u.at((0.5, 1.0)) == approx([1.0, 2.0])
    DirichletBC(V, (3.0, 4.0), 3).apply(u)
    assert u.at((0.5, 0.0)) == approx([3.0, 4.0])
    DirichletBC(V, as_vector((SpatialCoordinate(m)[0], 0.0)), 1).apply(u)
    assert u.at((0.0, 0.5)) == approx([0.0, 0.0])


def test_dirichlet_bc_vector_scalar_number_broadcasts():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V, 0.0, "on_boundary").apply(u)
    assert np.abs(u.dat.data).max() == 0.0


def test_dirichlet_bc_on_vector_component():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V.sub(1), 3.0, 1).apply(u)
    assert u.at((0.0, 0.5)) == approx([0.0, 3.0])
    DirichletBC(V.sub(0), 2.0, 2).apply(u)
    assert u.at((1.0, 0.5)) == approx([2.0, 0.0])


def test_dirichlet_bc_on_mixed_subspace():
    m = UnitSquareMesh(4, 4)
    Z = VectorFunctionSpace(m, "CG", 2) * FunctionSpace(m, "CG", 1)
    w = Function(Z)
    DirichletBC(Z.sub(0), Constant((1.0, 2.0)), 1).apply(w)
    DirichletBC(Z.sub(1), 5.0, 2).apply(w)
    u, p = w.subfunctions
    assert u.at((0.0, 0.5)) == approx([1.0, 2.0])
    assert p.at((1.0, 0.5)) == pytest.approx(5.0)
    assert p.at((0.0, 0.5)) == 0.0


def test_dirichlet_bc_on_mixed_component_of_vector_subspace():
    m = UnitSquareMesh(4, 4)
    Z = VectorFunctionSpace(m, "CG", 2) * FunctionSpace(m, "CG", 1)
    w = Function(Z)
    DirichletBC(Z.sub(0).sub(1), 7.0, 3).apply(w)
    assert w.subfunctions[0].at((0.5, 0.0)) == approx([0.0, 7.0])


def test_dirichlet_bc_sub_domain_attribute_and_space():
    V = FunctionSpace(UnitSquareMesh(2, 2), "CG", 1)
    bc = DirichletBC(V, 0.0, (1, 2))
    assert bc.sub_domain == (1, 2)
    assert bc.function_space() is V


def test_dirichlet_bc_by_name_extra():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V, 1.0, "left").apply(u)
    assert u.at((0.0, 0.3)) == pytest.approx(1.0)
    assert u.at((1.0, 0.3)) == 0.0


def test_dirichlet_bc_wrong_id_raises():
    V = FunctionSpace(UnitSquareMesh(2, 2), "CG", 1)
    with pytest.raises(ValueError):
        DirichletBC(V, 0.0, 9)


def test_dirichlet_bc_3d_ids():
    m = UnitCubeMesh(2, 2, 2)
    V = FunctionSpace(m, "CG", 1)
    u = Function(V)
    DirichletBC(V, 1.0, 6).apply(u)
    assert u.at((0.3, 0.4, 1.0)) == pytest.approx(1.0)
    assert u.at((0.3, 0.4, 0.5)) == 0.0
    assert int(round(u.dat.data.sum())) == 9


def test_dirichlet_bc_quad_mesh():
    m = UnitSquareMesh(4, 4, quadrilateral=True)
    V = FunctionSpace(m, "CG", 2)
    x, y = SpatialCoordinate(m)
    u = Function(V)
    DirichletBC(V, x + y, 4).apply(u)
    assert u.at((0.3, 1.0)) == pytest.approx(1.3)


def test_dirichlet_bc_dg_apply_does_not_crash():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "CG", 1)
    u = Function(V)
    bc = DirichletBC(V, 1.0, "on_boundary")
    bc.apply(u)
    bc.apply(u)
    assert int(round(u.dat.data.sum())) == 8


@xfail("DirichletBC has no function_arg / homogenize / reconstruct / nodes")
def test_dirichlet_bc_methods():
    V = FunctionSpace(UnitSquareMesh(2, 2), "CG", 1)
    bc = DirichletBC(V, 3.0, 1)
    bc.homogenize()
    assert len(bc.nodes) == 3


@xfail("DirichletBC.function_arg is missing; Firedrake lets time dependent scripts update g")
def test_dirichlet_bc_function_arg_update():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    g = Constant(1.0)
    bc = DirichletBC(V, g, 1)
    assert bc.function_arg is g


def test_dirichlet_bc_constant_update_before_apply():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    g = Constant(1.0)
    bc = DirichletBC(V, g, 1)
    u = Function(V)
    bc.apply(u)
    assert u.at((0.0, 0.5)) == pytest.approx(1.0)
    g.assign(4.0)
    bc.apply(u)
    assert u.at((0.0, 0.5)) == pytest.approx(4.0)


# ---------------------------------------------------------------- norms

def test_norm_types_on_scalar():
    m = UnitSquareMesh(8, 8)
    V = FunctionSpace(m, "CG", 1)
    x, y = SpatialCoordinate(m)
    u = Function(V).interpolate(x)
    assert norm(u) == pytest.approx(math.sqrt(1 / 3), rel=1e-12)
    assert norm(u, "L2") == pytest.approx(norm(u))
    assert norm(u, norm_type="l2") == pytest.approx(norm(u))
    assert norm(u, "H1") == pytest.approx(math.sqrt(1 / 3 + 1), rel=1e-12)
    assert norm(u, "L1") == pytest.approx(0.5, rel=1e-12)


def test_norm_lp_values():
    m = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(m)
    assert norm(x, "L4") == pytest.approx(0.2 ** 0.25, rel=1e-12)
    assert norm(x, "L3") == pytest.approx(0.25 ** (1 / 3), rel=1e-12)


def test_norm_of_expression_difference():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 2)
    x, y = SpatialCoordinate(m)
    u = Function(V).interpolate(x * x)
    w = Function(V).interpolate(x * x + 0.5)
    assert norm(u - w) == pytest.approx(0.5, rel=1e-12)
    assert norm(w - u, "H1") == pytest.approx(0.5, rel=1e-12)


def test_norm_vector_hdiv_hcurl():
    m = UnitSquareMesh(8, 8)
    x, y = SpatialCoordinate(m)
    v = as_vector((x, y))
    assert norm(v) == pytest.approx(math.sqrt(2 / 3), rel=1e-12)
    assert norm(v, "Hdiv") == pytest.approx(math.sqrt(2 / 3 + 4), rel=1e-12)
    w = as_vector((-y, x))
    assert norm(w, "Hcurl") == pytest.approx(math.sqrt(2 / 3 + 4), rel=1e-12)
    assert norm(v, "H1") == pytest.approx(math.sqrt(2 / 3 + 2), rel=1e-12)


def test_norm_of_vector_function():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "CG", 1)
    u = Function(V).interpolate(as_vector((1.0, 2.0)))
    assert norm(u) == pytest.approx(math.sqrt(5.0))


def test_norm_unknown_type_raises():
    m = UnitSquareMesh(2, 2)
    with pytest.raises((ValueError, RuntimeError)):
        norm(Constant(1.0) * SpatialCoordinate(m)[0], "bogus")


@xfail("norm() raises ValueError for an unknown type, Firedrake raises RuntimeError")
def test_norm_unknown_type_is_runtime_error():
    m = UnitSquareMesh(2, 2)
    with pytest.raises(RuntimeError):
        norm(SpatialCoordinate(m)[0], "bogus")


def test_errornorm_expression_vs_function():
    m = UnitSquareMesh(8, 8)
    V = FunctionSpace(m, "CG", 2)
    x, y = SpatialCoordinate(m)
    uh = Function(V).interpolate(x * x)
    assert errornorm(x * x, uh) == pytest.approx(0.0, abs=1e-12)
    assert errornorm(x * x + 1, uh) == pytest.approx(1.0, rel=1e-12)
    assert errornorm(x * x + 1, uh, "H1") == pytest.approx(math.sqrt(1.0), rel=1e-12)
    assert errornorm(x * x + 1, uh, norm_type="L2") == pytest.approx(1.0, rel=1e-12)


def test_errornorm_function_vs_function():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    a = Function(V).assign(1.0)
    b = Function(V).assign(3.0)
    assert errornorm(a, b) == pytest.approx(2.0)
    assert errornorm(a, b, "H1") == pytest.approx(2.0)


def test_errornorm_convergence_rates():
    errs = []
    for n in (4, 8, 16):
        m = UnitSquareMesh(n, n)
        V = FunctionSpace(m, "CG", 1)
        x, y = SpatialCoordinate(m)
        ex = sin(pi * x) * sin(pi * y)
        uh = Function(V).interpolate(ex)
        errs.append(errornorm(ex, uh))
    assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.5


def test_errornorm_with_degree_rise_ignored():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "CG", 1)
    uh = Function(V)
    assert errornorm(Constant(1.0) + 0 * SpatialCoordinate(m)[0], uh, degree_rise=2) == pytest.approx(1.0)


def test_norm_on_1d_and_3d():
    m1 = UnitIntervalMesh(10)
    x1 = SpatialCoordinate(m1)[0]
    assert norm(x1) == pytest.approx(math.sqrt(1 / 3), rel=1e-12)
    m3 = UnitCubeMesh(2, 2, 2)
    assert norm(SpatialCoordinate(m3)[0]) == pytest.approx(math.sqrt(1 / 3), rel=1e-12)


def test_norm_hdiv_function_rt():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "RT", 1)
    x, y = SpatialCoordinate(m)
    u = Function(V).interpolate(as_vector((x, y)))
    assert norm(u, "Hdiv") == pytest.approx(math.sqrt(2 / 3 + 4), rel=1e-10)


def test_norm_hcurl_function_n1curl():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "N1curl", 1)
    x, y = SpatialCoordinate(m)
    u = Function(V).interpolate(as_vector((-y, x)))
    assert norm(u, "Hcurl") == pytest.approx(math.sqrt(2 / 3 + 4), rel=1e-10)


def test_mesh_dimension_properties():
    m = UnitSquareMesh(2, 2)
    assert m.geometric_dimension + 1 == 3
    assert list(range(m.topological_dimension)) == [0, 1]


# ---------------------------------------------------------------- demos and small problems

def test_helmholtz_demo():
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(cos(x * pi * 2) * cos(y * pi * 2))
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u = Function(V)
    solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})
    f.interpolate(cos(x * pi * 2) * cos(y * pi * 2) / (1 + 8 * pi * pi))
    err = sqrt(assemble(dot(u - f, u - f) * dx))
    assert err < 2e-3


def test_helmholtz_demo_second_order_convergence():
    errs = []
    for n in (8, 16):
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, "CG", 1)
        x, y = SpatialCoordinate(mesh)
        ex = cos(2 * pi * x) * cos(2 * pi * y)
        f = Function(V).interpolate(ex)
        u = TrialFunction(V)
        v = TestFunction(V)
        uh = Function(V)
        solve((inner(grad(u), grad(v)) + inner(u, v)) * dx == inner(f, v) * dx, uh,
              solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})
        errs.append(errornorm(ex / (1 + 8 * pi * pi), uh))
    assert 3.2 < errs[0] / errs[1] < 4.8


@pytest.mark.parametrize("params", [
    {}, {'ksp_type': 'preonly', 'pc_type': 'lu'}, {'ksp_type': 'cg', 'pc_type': 'none'},
    {'ksp_type': 'cg', 'pc_type': 'jacobi', 'ksp_rtol': 1e-12},
    {'ksp_type': 'gmres', 'pc_type': 'none', 'ksp_rtol': 1e-12, 'ksp_max_it': 500},
    {'ksp_type': 'preonly', 'pc_type': 'cholesky'},
    {'ksp_type': 'cg', 'pc_type': 'lu'},
    {'mat_type': 'aij', 'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
], ids=lambda p: "-".join(f"{k}={v}" for k, v in p.items()) or "default")
def test_solver_parameter_variants_agree(params):
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    f = x * y
    a = inner(grad(u), grad(v)) * dx + u * v * dx
    solve(a == f * v * dx, uh, solver_parameters=params)
    ref = Function(V)
    solve(a == f * v * dx, ref)
    assert assemble((uh - ref) ** 2 * dx) < 1e-16


def test_ksp_monitor_option_runs():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(u * v * dx == Constant(1.0) * v * dx, uh,
          solver_parameters={'ksp_type': 'cg', 'pc_type': 'none', 'ksp_monitor': None})
    assert assemble(uh * dx) == pytest.approx(1.0, rel=1e-6)


def test_poisson_dirichlet_ids_and_neumann():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    ex = 1 + x * x + 2 * y * y
    f = Constant(-6.0)
    g = 4 * y
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds(4)
    bcs = [DirichletBC(V, ex, (1, 2, 3))]
    uh = Function(V)
    solve(a == L, uh, bcs=bcs)
    assert errornorm(ex, uh) < 1e-12
    assert errornorm(ex, uh, "H1") < 1e-11


def test_poisson_neumann_p1_converges():
    errs = []
    for n in (4, 8, 16):
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, "CG", 1)
        x, y = SpatialCoordinate(mesh)
        ex = sin(pi * x) * cos(pi * y)
        f = 2 * pi ** 2 * ex
        gN = -pi * sin(pi * x) * sin(pi * y)
        u, v = TrialFunction(V), TestFunction(V)
        L = f * v * dx + gN * v * ds(4)
        bc = [DirichletBC(V, ex, 1), DirichletBC(V, ex, 2), DirichletBC(V, ex, 3)]
        uh = Function(V)
        solve(inner(grad(u), grad(v)) * dx == L, uh, bcs=bc)
        errs.append(errornorm(ex, uh))
    assert errs[0] / errs[1] > 3.3 and errs[1] / errs[2] > 3.6


def test_poisson_quads_p2_exact():
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    ex = 1 + x * x + y
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == Constant(-2.0) * v * dx, uh, bcs=DirichletBC(V, ex, "on_boundary"))
    assert errornorm(ex, uh) < 1e-12


def test_poisson_3d_tets():
    mesh = UnitCubeMesh(3, 3, 3)
    V = FunctionSpace(mesh, "CG", 2)
    x, y, z = SpatialCoordinate(mesh)
    ex = x * x + y * z
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == Constant(-2.0) * v * dx, uh,
          bcs=[DirichletBC(V, ex, "on_boundary")])
    assert errornorm(ex, uh) < 1e-11


def test_poisson_hex_boundary_ids():
    mesh = UnitCubeMesh(2, 2, 2, hexahedral=True)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == Constant(0.0) * v * dx, uh,
          bcs=[DirichletBC(V, 1.0, 5), DirichletBC(V, 3.0, 6)])
    assert uh.at((0.3, 0.4, 0.25)) == pytest.approx(1.5)
    assert uh.at((0.3, 0.4, 0.75)) == pytest.approx(2.5)


def test_poisson_1d():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)
    x = SpatialCoordinate(mesh)[0]
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == Constant(2.0) * v * dx, uh,
          bcs=[DirichletBC(V, 0.0, (1, 2))])
    assert errornorm(x * (1 - x), uh) < 1e-12


def test_poisson_robin_term_on_ds():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    a = inner(grad(u), grad(v)) * dx + u * v * ds(2)
    L = Constant(1.0) * v * ds(2)
    solve(a == L, uh, bcs=DirichletBC(V, 0.0, 1))
    assert uh.at((0.5, 0.5)) == pytest.approx(0.25, abs=1e-12)
    assert uh.at((1.0, 0.5)) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("fam,deg", [("BDM", 1), ("RT", 1), ("BDM", 2), ("RT", 2)])
def test_mixed_poisson_rt_bdm_dg(fam, deg):
    sig_errs, u_errs = [], []
    for n in (4, 8):
        mesh = UnitSquareMesh(n, n)
        S = FunctionSpace(mesh, fam, deg)
        Q = FunctionSpace(mesh, "DG", deg - 1)
        W = S * Q
        x, y = SpatialCoordinate(mesh)
        ex = sin(pi * x) * sin(pi * y)
        f = 2 * pi ** 2 * ex
        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)
        a = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
        L = -f * v * dx
        w = Function(W)
        solve(a == L, w)
        sh, uh = w.subfunctions
        u_errs.append(errornorm(ex, uh))
        sig_errs.append(errornorm(grad(ex), sh))
    assert u_errs[0] / u_errs[1] > 1.8
    assert sig_errs[0] / sig_errs[1] > 1.8


def test_mixed_poisson_demo_with_dirichlet_flux_bcs():
    mesh = UnitSquareMesh(8, 8)
    BDM = FunctionSpace(mesh, "BDM", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    W = BDM * DG
    x, y = SpatialCoordinate(mesh)
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    f = Function(DG).interpolate(10 * exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))
    a = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
    L = -f * v * dx
    bc0 = DirichletBC(W.sub(0), as_vector([0.0, -sin(5 * x)]), 3)
    bc1 = DirichletBC(W.sub(0), as_vector([0.0, sin(5 * x)]), 4)
    w = Function(W)
    solve(a == L, w, bcs=[bc0, bc1])
    sigma_h, u_h = w.subfunctions
    n = FacetNormal(mesh)
    assert assemble(dot(sigma_h, n) * ds(4)) == pytest.approx(assemble(sin(5 * x) * ds(4)), rel=1e-8)
    assert assemble(dot(sigma_h, n) * ds(3)) == pytest.approx(assemble(sin(5 * x) * ds(3)), rel=1e-8)


def stokes_exact(mesh):
    x, y = SpatialCoordinate(mesh)
    ue = as_vector((x**2 * (1 - x)**2 * (2 * y * (1 - y)**2 - 2 * y**2 * (1 - y)),
                    -(2 * x * (1 - x)**2 - 2 * x**2 * (1 - x)) * y**2 * (1 - y)**2))
    return ue, x - 0.5


def test_stokes_taylor_hood_manufactured_with_nullspace():
    mesh = UnitSquareMesh(8, 8)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    ue, pe = stokes_exact(mesh)
    f = -div(grad(ue)) + grad(pe)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = (inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx
    L = inner(f, v) * dx
    bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary")]
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
    w = Function(W)
    solve(a == L, w, bcs=bcs, nullspace=nullspace)
    uh, ph = w.subfunctions
    pmean = assemble(ph * dx)
    assert errornorm(ue, uh) < 3e-3
    assert math.sqrt(assemble((ph - pmean - pe) ** 2 * dx)) < 5e-2
    assert math.sqrt(assemble(div(uh) ** 2 * dx)) < 2e-2


def test_stokes_lid_driven_cavity_demo():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W
    u, p = TrialFunctions(Z)
    v, q = TestFunctions(Z)
    a = (inner(grad(u), grad(v)) - p * div(v) + div(u) * q) * dx
    L = inner(Constant((0.0, 0.0)), v) * dx
    bcs = [DirichletBC(Z.sub(0), Constant((1.0, 0.0)), 4),
           DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 2, 3))]
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
    z = Function(Z)
    solve(a == L, z, bcs=bcs, nullspace=nullspace)
    u_h, p_h = z.subfunctions
    assert u_h.at((0.5, 1.0)) == approx([1.0, 0.0], abs=1e-12)
    assert u_h.at((0.0, 0.5)) == approx([0.0, 0.0], abs=1e-12)
    assert u_h.at((0.5, 0.9))[0] > 0.3
    assert u_h.at((0.5, 0.3))[0] < 0.0
    assert abs(assemble(dot(u_h, FacetNormal(mesh)) * ds)) < 1e-10
    assert abs(assemble(div(u_h) * dx)) < 1e-10


def test_stokes_minres_options():
    mesh = UnitSquareMesh(6, 6)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    ue, pe = stokes_exact(mesh)
    f = -div(grad(ue)) + grad(pe)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = (inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
    bcs = DirichletBC(W.sub(0), (0.0, 0.0), "on_boundary")
    w1, w2 = Function(W), Function(W)
    solve(a == inner(f, v) * dx, w1, bcs=bcs, nullspace=nullspace)
    solve(a == inner(f, v) * dx, w2, bcs=bcs, nullspace=nullspace,
          solver_parameters={'ksp_type': 'gmres', 'pc_type': 'lu', 'ksp_rtol': 1e-12})
    assert errornorm(w1.subfunctions[0], w2.subfunctions[0]) < 1e-8


def test_pure_neumann_poisson_with_constant_nullspace():
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    ex = cos(pi * x) * cos(pi * y)
    f = Function(V).interpolate(2 * pi**2 * ex)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == f * v * dx, uh, nullspace=VectorSpaceBasis(constant=True))
    mean = assemble(uh * dx)
    assert math.sqrt(assemble((uh - mean - ex) ** 2 * dx)) < 1e-2


def test_pure_neumann_with_assembled_forms_and_nullspace():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    ex = cos(pi * x)
    f = pi**2 * ex
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(inner(grad(u), grad(v)) * dx)
    b = assemble(f * v * dx)
    uh = Function(V)
    solve(A, uh, b, nullspace=VectorSpaceBasis(constant=True))
    mean = assemble(uh * dx)
    assert math.sqrt(assemble((uh - mean - ex) ** 2 * dx)) < 2e-2


def test_nonlinear_poisson_F_eq_0():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    ex = 1 + x + 2 * y
    f = -10 * ex
    u = Function(V)
    v = TestFunction(V)
    F = inner((1 + u**2) * grad(u), grad(v)) * dx - f * v * dx
    u.assign(1.0)
    solve(F == 0, u, bcs=DirichletBC(V, ex, "on_boundary"))
    assert errornorm(ex, u) < 1e-9


def test_nonlinear_poisson_with_explicit_jacobian():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    ex = 1 + x + 2 * y
    f = -10 * ex
    u = Function(V).assign(1.0)
    v = TestFunction(V)
    F = inner((1 + u**2) * grad(u), grad(v)) * dx - f * v * dx
    J = derivative(F, u)
    solve(F == 0, u, bcs=[DirichletBC(V, ex, "on_boundary")], J=J)
    assert errornorm(ex, u) < 1e-9


def test_nonlinear_poisson_rhs_form_equation():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    ex = 1 + x + 2 * y
    u = Function(V).assign(1.0)
    v = TestFunction(V)
    solve(inner((1 + u**2) * grad(u), grad(v)) * dx == -10 * ex * v * dx, u,
          bcs=DirichletBC(V, ex, "on_boundary"))
    assert errornorm(ex, u) < 1e-9


def test_nonlinear_solver_parameters():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V).assign(1.0)
    v = TestFunction(V)
    F = (u * u - 2.0) * v * dx
    solve(F == 0, u, solver_parameters={'snes_rtol': 1e-12, 'snes_atol': 1e-14, 'snes_max_it': 30,
                                        'snes_monitor': None})
    assert u.at((0.3, 0.3)) == pytest.approx(math.sqrt(2.0), rel=1e-6)


def test_nonlinear_variational_solver_options_and_convergence_error():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V).assign(0.1)
    v = TestFunction(V)
    F = (u * u * u - 8.0) * v * dx
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters={'snes_max_it': 2, 'snes_rtol': 1e-14,
                                                                    'snes_atol': 1e-14})
    with pytest.raises(ConvergenceError):
        solver.solve()
    u.assign(1.0)
    solver = NonlinearVariationalSolver(problem, solver_parameters={'snes_rtol': 1e-12, 'snes_max_it': 50})
    solver.solve()
    assert u.at((0.5, 0.5)) == pytest.approx(2.0, rel=1e-8)


def test_nonlinear_variational_solver_reuse_with_new_constant():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    c = Constant(2.0)
    u = Function(V).assign(1.0)
    v = TestFunction(V)
    problem = NonlinearVariationalProblem((u * u - c) * v * dx, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters={'snes_rtol': 1e-12})
    solver.solve()
    assert u.at((0.4, 0.4)) == pytest.approx(math.sqrt(2.0), rel=1e-6)
    c.assign(9.0)
    u.assign(1.0)
    solver.solve()
    assert u.at((0.4, 0.4)) == pytest.approx(3.0, rel=1e-6)


def test_nonlinear_problem_with_J_and_bcs_1d():
    mesh = UnitIntervalMesh(20)
    V = FunctionSpace(mesh, "CG", 2)
    x = SpatialCoordinate(mesh)[0]
    u = Function(V).interpolate(x)
    v = TestFunction(V)
    ex = x * (1 - x)
    f = 2 + ex**3
    F = (inner(grad(u), grad(v)) + u**3 * v - f * v) * dx
    problem = NonlinearVariationalProblem(F, u, bcs=DirichletBC(V, 0.0, (1, 2)), J=derivative(F, u))
    NonlinearVariationalSolver(problem).solve()
    assert errornorm(ex, u) < 1e-4


def test_burgers_time_stepping_with_assign():
    n = 30
    mesh = UnitIntervalMesh(n)
    V = VectorFunctionSpace(mesh, "CG", 2)
    u_ = Function(V, name="Velocity")
    u = Function(V, name="NextVelocity")
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    ic = project(as_vector([sin(2 * pi * x[0])]), V)
    u_.assign(ic)
    u.assign(ic)
    nu = 0.0001
    timestep = 1.0 / n
    F = (inner((u - u_) / timestep, v) + inner(dot(u, nabla_grad(u)), v) + nu * inner(grad(u), grad(v))) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    energies = [assemble(inner(u_, u_) * dx)]
    for step in range(8):
        solve(F == 0, u, bcs=bc)
        u_.assign(u)
        energies.append(assemble(inner(u_, u_) * dx))
    assert all(b < a for a, b in zip(energies, energies[1:]))
    assert abs(u_.at(0.25)[0] + u_.at(0.75)[0]) < 1e-8
    assert u_.at(0.0)[0] == pytest.approx(0.0, abs=1e-12)
    assert 0.3 < u_.at(0.25)[0] < 1.0


def test_dg_advection_steady_upwind():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "DG", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    velocity = Function(W).interpolate(as_vector((1.0, 0.0)))
    n = FacetNormal(mesh)
    un = 0.5 * (dot(velocity, n) + abs(dot(velocity, n)))
    q, v = TrialFunction(V), TestFunction(V)
    q_in = Function(V).interpolate(y)
    a = (-q * dot(velocity, grad(v)) * dx
         + dot(jump(v), un('+') * q('+') - un('-') * q('-')) * dS
         + dot(v, un * q) * ds)
    L = -v * (dot(velocity, n) - un) * q_in * ds
    qh = Function(V)
    solve(a == L, qh)
    assert errornorm(y, qh) < 1e-10


def test_dg_advection_time_loop_linear_solver():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "DG", 1)
    x, y = SpatialCoordinate(mesh)
    c = as_vector((1.0, 0.0))
    n = FacetNormal(mesh)
    cn = 0.5 * (dot(c, n) + abs(dot(c, n)))
    q, v = TrialFunction(V), TestFunction(V)
    dt = 0.01
    q_old = Function(V).interpolate(y)
    adv = (-q * dot(c, grad(v)) * dx
           + dot(jump(v), cn('+') * q('+') - cn('-') * q('-')) * dS
           + v * cn * q * ds)
    inflow = v * (dot(c, n) - cn) * y * ds
    qn = Function(V)
    problem = LinearVariationalProblem(q * v * dx + dt * adv, q_old * v * dx - dt * inflow, qn)
    solver = LinearVariationalSolver(problem)
    for _ in range(5):
        solver.solve()
        q_old.assign(qn)
    assert errornorm(y, q_old) < 1e-8


def sipg(mesh, k, ex, f, alpha=10.0):
    V = FunctionSpace(mesh, "DG", k)
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg = (h('+') + h('-')) / 2
    u, v = TrialFunction(V), TestFunction(V)
    a = (inner(grad(u), grad(v)) * dx
         - inner(avg(grad(u)), jump(v, n)) * dS
         - inner(jump(u, n), avg(grad(v))) * dS
         + alpha / h_avg * inner(jump(u, n), jump(v, n)) * dS
         - inner(grad(u), n) * v * ds
         - u * inner(grad(v), n) * ds
         + alpha / h * u * v * ds)
    L = (f * v * dx
         - ex * inner(grad(v), n) * ds
         + alpha / h * ex * v * ds)
    uh = Function(V)
    solve(a == L, uh)
    return uh


def test_interior_penalty_poisson_exact_for_p2():
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    ex = x * x + y * y
    uh = sipg(mesh, 2, ex, Constant(-4.0))
    assert errornorm(ex, uh) < 1e-10


def test_interior_penalty_poisson_convergence_dg1():
    errs = []
    for n in (4, 8, 16):
        mesh = UnitSquareMesh(n, n)
        x, y = SpatialCoordinate(mesh)
        ex = sin(pi * x) * sin(pi * y)
        errs.append(errornorm(ex, sipg(mesh, 1, ex, 2 * pi**2 * ex)))
    assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.5


def test_interior_facet_forms_jump_avg_values():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "DG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(V).interpolate(x + 2 * y)
    assert assemble(jump(u) ** 2 * dS) == pytest.approx(0.0, abs=1e-24)
    assert assemble(avg(u) * dS) == pytest.approx(assemble(u('+') * dS))
    z = Function(V).interpolate(conditional(lt(x, 0.5), 0.0, 1.0))
    assert assemble(sqrt(jump(z) ** 2) * dS) > 0.5


@pytest.mark.parametrize("make", [
    lambda z: abs(jump(z)), lambda z: sign(jump(z)) * jump(z),
    lambda z: max_value(jump(z), -jump(z)), lambda z: conditional(gt(jump(z), 0), jump(z), -jump(z))],
    ids=["abs", "sign", "max_value", "conditional"])
def test_nonsmooth_functions_of_jumps_on_dS(make):
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    chi = Function(FunctionSpace(mesh, "DG", 0)).interpolate(conditional(lt(x, 0.5), 0.0, 1.0))
    z = Function(FunctionSpace(mesh, "DG", 1)).interpolate(chi * (1 + y))
    # the jump is 1 + y across x = 0.5 and vanishes elsewhere
    assert assemble(make(z) * dS) == pytest.approx(1.5)


def test_linear_elasticity_3d_ids_uniaxial():
    eps = 0.05
    Lx = 2.0
    mesh = BoxMesh(3, 2, 2, Lx, 1.0, 1.0)
    V = VectorFunctionSpace(mesh, "CG", 1)
    E, nu = 10.0, 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return 2 * mu * epsilon(w) + lmbda * tr(epsilon(w)) * Identity(3)

    u, v = TrialFunction(V), TestFunction(V)
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(Constant((0.0, 0.0, 0.0)), v) * dx
    bcs = [DirichletBC(V.sub(0), 0.0, 1), DirichletBC(V.sub(0), eps * Lx, 2),
           DirichletBC(V.sub(1), 0.0, 3), DirichletBC(V.sub(2), 0.0, 5)]
    uh = Function(V)
    solve(a == L, uh, bcs=bcs)
    x, y, z = SpatialCoordinate(mesh)
    ex = as_vector((eps * x, -nu * eps * y, -nu * eps * z))
    assert errornorm(ex, uh) < 1e-10
    assert assemble(sigma(uh)[0, 0] * dx) == pytest.approx(E * eps * Lx, rel=1e-10)
    assert assemble(sigma(uh)[1, 1] * dx) == pytest.approx(0.0, abs=1e-10)


def test_linear_elasticity_cantilever_demo():
    mesh = BoxMesh(10, 2, 2, 5.0, 1.0, 1.0)
    V = VectorFunctionSpace(mesh, "CG", 1)
    mu, lmbda, rho, g = 1.0, 1.25, 1.0, 0.1

    def epsilon(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

    def sigma(u):
        return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)

    u, v = TrialFunction(V), TestFunction(V)
    f = Constant((0.0, 0.0, -rho * g))
    a = inner(sigma(u), epsilon(v)) * dx
    L = dot(f, v) * dx
    bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), 1)
    uh = Function(V)
    solve(a == L, uh, bcs=bc, solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi', 'ksp_rtol': 1e-10,
                                                 'ksp_max_it': 2000})
    tip = uh.at((5.0, 0.5, 0.5))
    assert tip[2] < -0.01
    assert abs(tip[1]) < abs(tip[2])
    assert uh.at((0.0, 0.5, 0.5)) == approx([0.0, 0.0, 0.0], abs=1e-12)


def test_assembled_matrix_workflow():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u, v = TrialFunction(V), TestFunction(V)
    f = Function(V).interpolate(x * y)
    a = inner(grad(u), grad(v)) * dx + u * v * dx
    L = f * v * dx
    bcs = [DirichletBC(V, 1.0, 1), DirichletBC(V, 0.0, 2)]
    A = assemble(a, bcs=bcs)
    b = assemble(L)
    for bc in bcs:
        bc.apply(b)
    uh = Function(V)
    solve(A, uh, b)
    ref = Function(V)
    solve(a == L, ref, bcs=bcs)
    assert assemble((uh - ref) ** 2 * dx) < 1e-20
    assert uh.at((0.0, 0.3)) == pytest.approx(1.0)
    assert uh.at((1.0, 0.3)) == pytest.approx(0.0)


def test_assembled_matrix_workflow_with_solver_parameters():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    bc = DirichletBC(V, 2.0, "on_boundary")
    A = assemble(inner(grad(u), grad(v)) * dx, bcs=bc)
    b = assemble(Constant(0.0) * v * dx)
    bc.apply(b)
    uh = Function(V)
    solve(A, uh, b, solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi', 'ksp_rtol': 1e-12})
    assert uh.at((0.5, 0.5)) == pytest.approx(2.0, rel=1e-9)


def test_linear_solver_class_reuse():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    bc = DirichletBC(V, 0.0, "on_boundary")
    A = assemble(inner(grad(u), grad(v)) * dx, bcs=bc)
    solver = LinearSolver(A, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    x, y = SpatialCoordinate(mesh)
    first = None
    for scale in (1.0, 2.0):
        b = assemble(scale * x * v * dx)
        bc.apply(b)
        uh = Function(V)
        solver.solve(uh, b)
        if first is None:
            first = uh.copy(deepcopy=True)
        else:
            assert assemble((uh - 2 * first) ** 2 * dx) < 1e-20


def test_linear_variational_solver_in_time_loop_heat():
    mesh = UnitSquareMesh(12, 12)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    u_old = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    u_new = Function(V)
    u, v = TrialFunction(V), TestFunction(V)
    dt = 0.005
    a = u * v * dx + dt * inner(grad(u), grad(v)) * dx
    L = u_old * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    problem = LinearVariationalProblem(a, L, u_new, bcs=bc, constant_jacobian=True)
    solver = LinearVariationalSolver(problem, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    t = 0.0
    for _ in range(10):
        solver.solve()
        u_old.assign(u_new)
        t += dt
    be = (1 + 2 * pi**2 * dt) ** -10
    assert u_old.at((0.5, 0.5)) == pytest.approx(be, rel=1e-3)
    assert errornorm(be * sin(pi * x) * sin(pi * y), u_old) < 2e-3


def test_linear_variational_solver_constant_update():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    f = Constant(1.0)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    problem = LinearVariationalProblem(inner(grad(u), grad(v)) * dx, f * v * dx, uh,
                                       bcs=DirichletBC(V, 0.0, "on_boundary"))
    solver = LinearVariationalSolver(problem)
    solver.solve()
    first = uh.copy(deepcopy=True)
    f.assign(2.0)
    solver.solve()
    assert assemble((uh - 2 * first) ** 2 * dx) < 1e-20
    assert assemble(uh * dx) > assemble(first * dx) * 1.99


def test_linear_variational_solver_options_prefix_and_nullspace():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    problem = LinearVariationalProblem(inner(grad(u), grad(v)) * dx, pi**2 * cos(pi * x) * v * dx, uh)
    solver = LinearVariationalSolver(problem, nullspace=VectorSpaceBasis(constant=True),
                                     options_prefix="neumann_",
                                     solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi'})
    solver.solve()
    assert uh.at((0.0, 0.5)) > uh.at((1.0, 0.5))


def test_bc_value_updated_through_constant_in_loop():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    g = Constant(0.0)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    problem = LinearVariationalProblem(inner(grad(u), grad(v)) * dx, Constant(0.0) * v * dx, uh,
                                       bcs=[DirichletBC(V, g, 1), DirichletBC(V, 0.0, 2)])
    solver = LinearVariationalSolver(problem)
    for val in (1.0, 2.0, 3.0):
        g.assign(val)
        solver.solve()
        assert uh.at((0.5, 0.5)) == pytest.approx(val / 2, rel=1e-10)


def test_interpolate_with_constant_in_time_loop():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    t = Constant(0.0)
    f = Function(V)
    for k in range(3):
        t.assign(0.5 * k)
        f.interpolate(t * x * y)
        assert f.at((0.5, 0.5)) == pytest.approx(0.5 * k * 0.25)


def test_problem_with_function_coefficient_updated_between_solves():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    k = Function(V).assign(1.0)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    a = k * inner(grad(u), grad(v)) * dx
    L = Constant(1.0) * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(a == L, uh, bcs=bc)
    s1 = assemble(uh * dx)
    k.assign(2.0)
    solve(a == L, uh, bcs=bc)
    assert assemble(uh * dx) == pytest.approx(s1 / 2, rel=1e-10)


# ---------------------------------------------------------------- more UFL / form manipulation

def test_action_adjoint_lhs_rhs_system():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    w = Function(V).assign(1.0)
    a = u * v * dx
    assert assemble(action(a, w)).dat.data.sum() == pytest.approx(1.0)
    assert dense(assemble(adjoint(a)).mat) == approx(dense(assemble(a).mat).T)
    F = (u * v - Constant(2.0) * v) * dx
    assert assemble(rhs(F)).dat.data.sum() == pytest.approx(2.0)
    assert assemble(lhs(F)).mat.height == 25
    A, L = system(F)
    assert len(A.arguments()) == 2 and len(L.arguments()) == 1


def test_derivative_and_replace():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    w = Function(V).assign(2.0)
    v = TestFunction(V)
    dF = derivative(w**3 * dx, w, v)
    assert assemble(dF).dat.data.sum() == pytest.approx(12.0)
    J = derivative(derivative(w**3 * dx, w), w)
    assert assemble(J).mat.height == 25
    c = Constant(5.0)
    assert assemble(replace(w * dx, {w: c})) == pytest.approx(5.0)


def test_nabla_and_derivative_shortcuts():
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)
    u = Function(V).interpolate(x * y + y * y)
    assert assemble(u.dx(0) * dx) == pytest.approx(0.5, rel=1e-12)
    assert assemble(u.dx(1) * dx) == pytest.approx(1.5, rel=1e-12)
    assert assemble(div(grad(u)) * dx) == pytest.approx(2.0, rel=1e-10)
    w = as_vector((x, y))
    assert assemble(div(w) * dx) == pytest.approx(2.0)
    assert assemble(curl(as_vector((-y, x))) * dx) == pytest.approx(2.0)


def test_tensor_algebra_in_forms():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(V).interpolate(as_vector((x, 2 * y)))
    assert assemble(tr(grad(u)) * dx) == pytest.approx(3.0)
    assert assemble(det(Identity(2) + grad(u)) * dx) == pytest.approx(6.0)
    assert assemble(inner(sym(grad(u)), grad(u)) * dx) == pytest.approx(5.0)
    assert assemble(dev(grad(u))[0, 0] * dx) == pytest.approx(-0.5)
    assert assemble(inner(outer(u, u), Identity(2)) * dx) == pytest.approx(assemble(inner(u, u) * dx))
    assert assemble(inner(transpose(grad(u)), grad(u)) * dx) == pytest.approx(5.0)
    assert assemble(dot(grad(u), Constant((1.0, 1.0)))[1] * dx) == pytest.approx(2.0)


def test_measure_options_and_subdomain_ids():
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    assert assemble(x * ds(1) + x * ds(2)) == pytest.approx(1.0)
    assert assemble(Constant(1.0) * ds(1, domain=mesh) + Constant(1.0) * ds(2, domain=mesh)) == pytest.approx(2.0)
    assert assemble(x * dx(degree=1)) == pytest.approx(0.5)
    assert assemble(x * dx(mesh)) == pytest.approx(0.5)
    assert assemble(x * dx(domain=mesh, degree=3)) == pytest.approx(0.5)


# ---------------------------------------------------------------- pyplot

@pytest.fixture
def fields():
    plt.close("all")
    mesh = UnitSquareMesh(5, 5)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)
    u = Function(V, name="u").interpolate(x + 2 * y)
    W = VectorFunctionSpace(mesh, "CG", 1)
    w = Function(W, name="w").interpolate(as_vector((y, -x)))
    yield mesh, u, w
    plt.close("all")


def test_pyplot_module_all_names():
    for name in ("tripcolor", "tricontour", "tricontourf", "trisurf", "triplot", "quiver", "plot"):
        assert callable(getattr(fplt, name))


def test_tripcolor_returns_collection_and_values(fields):
    from matplotlib.collections import Collection
    mesh, u, w = fields
    fig, ax = plt.subplots()
    art = fplt.tripcolor(u, axes=ax)
    assert isinstance(art, Collection)
    assert art.axes is ax
    arr = np.asarray(art.get_array())
    assert arr.min() == pytest.approx(0.0, abs=1e-12)
    assert arr.max() == pytest.approx(3.0, abs=1e-12)


def test_tripcolor_without_axes_creates_figure(fields):
    mesh, u, w = fields
    n = len(plt.get_fignums())
    art = fplt.tripcolor(u)
    assert len(plt.get_fignums()) == n + 1
    assert art.axes.figure is not None


def test_tripcolor_keywords(fields):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    art = fplt.tripcolor(u, axes=ax, cmap="coolwarm", vmin=-1, vmax=4, num_sample_points=3)
    assert art.get_clim() == (-1, 4)
    assert art.get_cmap().name == "coolwarm"
    fig.colorbar(art, ax=ax)


def test_tripcolor_vector_field_shows_magnitude(fields):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    art = fplt.tripcolor(w, axes=ax)
    arr = np.asarray(art.get_array())
    assert arr.max() == pytest.approx(math.sqrt(2.0), rel=1e-6)
    assert arr.min() == pytest.approx(0.0, abs=1e-12)


def test_tripcolor_on_expression_via_interpolate(fields):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    art = fplt.tripcolor(assemble(interpolate(u * u, u.function_space())), axes=ax)
    assert np.asarray(art.get_array()).max() == pytest.approx(9.0, rel=1e-6)


def test_tripcolor_on_quad_and_dg(fields):
    mq = UnitSquareMesh(3, 3, quadrilateral=True)
    fig, ax = plt.subplots()
    fplt.tripcolor(Function(FunctionSpace(mq, "CG", 1)).assign(1.0), axes=ax)
    mesh, u, w = fields
    fplt.tripcolor(Function(FunctionSpace(mesh, "DG", 1)).assign(1.0), axes=ax)
    assert len(ax.collections) == 2


@pytest.mark.parametrize("name", ["tricontour", "tricontourf"])
def test_tricontour_variants(fields, name):
    from matplotlib.contour import ContourSet
    mesh, u, w = fields
    fig, ax = plt.subplots()
    cs = getattr(fplt, name)(u, axes=ax)
    assert isinstance(cs, ContourSet)
    cs2 = getattr(fplt, name)(u, 6, axes=ax)
    assert len(cs2.levels) >= 6
    cs3 = getattr(fplt, name)(u, levels=[0.5, 1.0, 2.0], axes=ax)
    assert list(cs3.levels)[:3] == [0.5, 1.0, 2.0] or len(cs3.levels) >= 3
    fig.colorbar(cs, ax=ax)


def test_tricontourf_vector_uses_magnitude(fields):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    cs = fplt.tricontourf(w, axes=ax)
    assert max(cs.levels) >= 1.3


def test_trisurf_3d_axes(fields):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh, u, w = fields
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    art = fplt.trisurf(u, axes=ax)
    assert isinstance(art, Poly3DCollection)
    assert art.axes is ax
    assert isinstance(fplt.trisurf(u), Poly3DCollection)


def test_triplot_boundary_labels(fields):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    arts = fplt.triplot(mesh, axes=ax)
    assert isinstance(arts, list) and len(arts) >= 5
    legend = ax.legend()
    assert sorted(t.get_text() for t in legend.get_texts()) == ["1", "2", "3", "4"]


def test_triplot_keywords_and_no_axes(fields):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    fplt.triplot(mesh, axes=ax, interior_kw={"linewidth": 0.1, "color": "r"},
                 boundary_kw={"linewidth": 3.0})
    n = len(plt.get_fignums())
    fplt.triplot(mesh)
    assert len(plt.get_fignums()) == n + 1


def test_triplot_quad_mesh():
    plt.close("all")
    mq = UnitSquareMesh(3, 3, quadrilateral=True)
    fig, ax = plt.subplots()
    fplt.triplot(mq, axes=ax)
    plt.close("all")


def test_quiver_values_and_returns_quiver(fields):
    from matplotlib.quiver import Quiver
    mesh, u, w = fields
    fig, ax = plt.subplots()
    q = fplt.quiver(w, axes=ax)
    assert isinstance(q, Quiver)
    xy = q.get_offsets()
    U = np.ma.getdata(q.U)
    Vv = np.ma.getdata(q.V)
    assert U == approx(xy[:, 1])
    assert Vv == approx(-xy[:, 0])
    q2 = fplt.quiver(w, axes=ax, scale=10, cmap="viridis")
    assert q2.scale == 10


def test_quiver_without_axes(fields):
    mesh, u, w = fields
    n = len(plt.get_fignums())
    fplt.quiver(w)
    assert len(plt.get_fignums()) == n + 1


def test_quiver_scalar_raises_valueerror(fields):
    mesh, u, w = fields
    with pytest.raises(ValueError):
        fplt.quiver(u)


def test_plot_1d_returns_lines(fields):
    plt.close("all")
    m = UnitIntervalMesh(10)
    x = SpatialCoordinate(m)[0]
    V = FunctionSpace(m, "CG", 2)
    u = Function(V, name="u").interpolate(x * x)
    fig, ax = plt.subplots()
    lines = fplt.plot(u, axes=ax)
    assert isinstance(lines, list) and len(lines) == 1
    xs, ys = lines[0].get_data()
    assert xs.min() == pytest.approx(0.0) and xs.max() == pytest.approx(1.0)
    assert np.allclose(ys, xs**2, atol=1e-12)
    assert np.all(np.diff(xs) >= 0)
    fplt.plot(u, "r--", axes=ax, linewidth=2)
    assert len(ax.lines) == 2


def test_plot_1d_without_axes_and_num_sample_points(fields):
    m = UnitIntervalMesh(4)
    u = Function(FunctionSpace(m, "CG", 1)).interpolate(SpatialCoordinate(m)[0])
    n = len(plt.get_fignums())
    lines = fplt.plot(u, num_sample_points=2)
    assert len(plt.get_fignums()) == n + 1
    assert len(lines[0].get_xdata()) >= 5


def test_plot_1d_dg_function_discontinuous():
    plt.close("all")
    m = UnitIntervalMesh(4)
    x = SpatialCoordinate(m)[0]
    u = Function(FunctionSpace(m, "DG", 0)).interpolate(conditional(lt(x, 0.5), 0.0, 1.0))
    fig, ax = plt.subplots()
    xs, ys = fplt.plot(u, axes=ax)[0].get_data()
    assert ys.min() == 0.0 and ys.max() == 1.0


@xfail("plot() lines are not labelled with the function name as in Firedrake")
def test_plot_1d_default_label(fields):
    m = UnitIntervalMesh(4)
    u = Function(FunctionSpace(m, "CG", 1), name="temperature")
    fig, ax = plt.subplots()
    line = fplt.plot(u, axes=ax)[0]
    assert line.get_label() == "temperature"


@xfail("plot() raises NotImplementedError for a 2D function, Firedrake raises ValueError")
def test_plot_2d_raises_valueerror(fields):
    mesh, u, w = fields
    with pytest.raises(ValueError):
        fplt.plot(u)


def test_pyplot_functions_do_not_show(fields):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    fplt.tripcolor(u, axes=ax)
    fplt.quiver(w, axes=ax)
    fplt.triplot(mesh, axes=ax)
    assert len(ax.collections) >= 3


def test_pyplot_savefig(fields, tmp_path):
    mesh, u, w = fields
    fig, ax = plt.subplots()
    fplt.tricontourf(u, axes=ax)
    out = tmp_path / "u.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000


# ---------------------------------------------------------------- VTKFile

def test_vtkfile_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u").interpolate(SpatialCoordinate(mesh)[0])
    f = VTKFile("out.pvd")
    for t in (0.0, 0.1, 0.2):
        f.write(u, time=t)
    assert (tmp_path / "out.pvd").exists()
    assert len(list(tmp_path.glob("out*.vtu"))) == 3


def test_vtkfile_multiple_functions_and_vector(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    u = Function(FunctionSpace(mesh, "CG", 1), name="p").interpolate(x)
    w = Function(VectorFunctionSpace(mesh, "CG", 2), name="vel").interpolate(as_vector((x, y)))
    f = VTKFile("multi.pvd")
    f.write(u, w, time=0.0)
    f.write(u, w, time=1.0)
    text = next(tmp_path.glob("multi*.vtu")).read_bytes()
    assert b"p" in text and b"vel" in text


def test_vtkfile_with_mixed_subfunctions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh = UnitSquareMesh(3, 3)
    Z = VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)
    z = Function(Z)
    uu, pp = z.subfunctions
    uu.rename("velocity")
    pp.rename("pressure")
    VTKFile("mixed.pvd").write(uu, pp)
    assert list(tmp_path.glob("mixed*.vtu"))


def test_vtkfile_absolute_path(tmp_path):
    mesh = UnitSquareMesh(3, 3)
    u = Function(FunctionSpace(mesh, "CG", 1), name="u")
    VTKFile(str(tmp_path / "abs.pvd")).write(u, time=0.0)
    assert list(tmp_path.glob("abs*"))


def test_vtkfile_1d_and_3d(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k, mesh in enumerate((UnitIntervalMesh(4), UnitCubeMesh(2, 2, 2))):
        u = Function(FunctionSpace(mesh, "CG", 1), name="u")
        VTKFile(f"m{k}.pvd").write(u)
    assert len(list(tmp_path.glob("m*.vtu"))) == 2


@xfail("VTKFile only writes a .pvd file when it is given times more than once; Firedrake always writes <name>.pvd")
def test_vtkfile_single_write_creates_pvd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh = UnitSquareMesh(2, 2)
    u = Function(FunctionSpace(mesh, "CG", 1), name="u")
    VTKFile("single.pvd").write(u)
    assert (tmp_path / "single.pvd").exists()


def test_vtkfile_firedrake_keywords(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    VTKFile("k.pvd", project_output=True, mode="w", comm=None)


def test_vtkfile_creates_directory(tmp_path):
    mesh = UnitSquareMesh(2, 2)
    u = Function(FunctionSpace(mesh, "CG", 1), name="u")
    VTKFile(str(tmp_path / "newdir" / "f.pvd")).write(u, time=0.0)
    assert any((tmp_path / "newdir").iterdir())


# ---------------------------------------------------------------- Firedrake features expected to be missing

@xfail("PeriodicUnitSquareMesh is not available")
def test_missing_periodic_mesh():
    PeriodicUnitSquareMesh(4, 4)


@xfail("ExtrudedMesh is not available")
def test_missing_extruded_mesh():
    ExtrudedMesh(UnitSquareMesh(2, 2), 3)


@xfail("UnitDiskMesh / other unstructured built-in meshes are not available")
def test_missing_unit_disk_mesh():
    UnitDiskMesh(2)


@xfail("Interpolator is not available")
def test_missing_interpolator():
    V = FunctionSpace(UnitSquareMesh(2, 2), "CG", 1)
    Interpolator(SpatialCoordinate(V.mesh())[0], V)


@xfail("VertexOnlyMesh / PointEvaluator are not available")
def test_missing_vertex_only_mesh_and_point_evaluator():
    m = UnitSquareMesh(2, 2)
    u = Function(FunctionSpace(m, "CG", 1)).assign(1.0)
    pe = PointEvaluator(m, [[0.3, 0.3]])
    assert pe.evaluate(u)[0] == pytest.approx(1.0)


@xfail("CheckpointFile / DumbCheckpoint are not available")
def test_missing_checkpoint_file(tmp_path):
    with CheckpointFile(str(tmp_path / "c.h5"), "w") as chk:
        chk.save_mesh(UnitSquareMesh(2, 2))


@xfail("Function.riesz_representation is missing")
def test_missing_riesz_representation():
    V = FunctionSpace(UnitSquareMesh(2, 2), "CG", 1)
    b = assemble(TestFunction(V) * dx)
    b.riesz_representation()


@xfail("firedrake.homogenize is missing")
def test_missing_homogenize():
    V = FunctionSpace(UnitSquareMesh(2, 2), "CG", 1)
    homogenize(DirichletBC(V, 1.0, 1))


@xfail("par_loop is missing")
def test_missing_par_loop():
    par_loop


@xfail("FacetArea is not supported in forms")
def test_missing_facet_area():
    m = UnitSquareMesh(2, 2)
    assert assemble(FacetArea(m) * ds) > 0


@xfail("MinFacetEdgeLength / MaxFacetEdgeLength are not supported in forms")
def test_missing_min_facet_edge_length():
    m = UnitSquareMesh(2, 2)
    assemble(MinFacetEdgeLength(m) * dx)


@xfail("mesh.coordinates is a copy: changing it does not move the mesh")
def test_moving_mesh_via_coordinates():
    m = UnitSquareMesh(4, 4)
    X = m.coordinates
    X.dat.data[:, 0] *= 2.0
    assert meas(m) == pytest.approx(2.0)


@xfail("mesh.coordinates.interpolate does not move the mesh")
def test_moving_mesh_via_coordinates_interpolate():
    m = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(m)
    m.coordinates.interpolate(as_vector((2 * x, y)))
    assert meas(m) == pytest.approx(2.0)


@xfail("mesh.comm / COMM_WORLD are None / missing")
def test_comm_world_and_mesh_comm():
    m = UnitSquareMesh(2, 2)
    assert COMM_WORLD.size == 1
    assert m.comm.rank == 0


@xfail("ngsufl.petsc.PETSc (PETSc.Print) is missing")
def test_petsc_print_import():
    from ngsufl.petsc import PETSc
    PETSc.Print("hello")


@xfail("Argument.function_space() is missing (Firedrake adds it to TestFunction/TrialFunction)")
def test_argument_function_space_method():
    V = FunctionSpace(UnitSquareMesh(2, 2), "CG", 1)
    assert TestFunction(V).function_space() is V


@xfail("Mesh.cell_sizes / exterior_facets / num_facets are missing")
def test_missing_mesh_accessors():
    m = UnitSquareMesh(2, 2)
    m.cell_sizes


@xfail("Function.dat.data for P2 etc. is not nodal; the Firedrake idiom u.dat.data[i] = point value fails")
def test_dat_data_p2_is_nodal():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "CG", 2)
    u = Function(V).interpolate(SpatialCoordinate(m)[0])
    assert sorted(set(np.round(u.dat.data_ro, 12))) == approx([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize("ksp,pc", [("gmres", "ilu"), ("cg", "hypre"), ("cg", "gamg"), ("cg", "sor"),
                                    ("bcgs", "jacobi"), ("gmres", "fieldsplit")])
@xfail("PETSc preconditioner / Krylov type is not available (NotImplementedError)")
def test_missing_petsc_options(ksp, pc):
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(u * v * dx == Constant(1.0) * v * dx, uh, solver_parameters={"ksp_type": ksp, "pc_type": pc})


def test_unsupported_petsc_option_is_explicit_error():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    with pytest.raises(NotImplementedError):
        solve(u * v * dx == Constant(1.0) * v * dx, Function(V),
              solver_parameters={"ksp_type": "gmres", "pc_type": "ilu"})


# ---------------------------------------------------------------- H(div), H(curl), CR, Real

@pytest.mark.parametrize("fam,deg,field", [
    ("RT", 1, lambda x, y: as_vector((x + 1, y - 2))),
    ("BDM", 1, lambda x, y: as_vector((x + 2 * y, 3 * x - y))),
    ("RT", 2, lambda x, y: as_vector((x * y, y * y))),
    ("N1curl", 1, lambda x, y: as_vector((1 - y, 2 + x))),
    ("N2curl", 1, lambda x, y: as_vector((x + y, x - 2 * y))),
])
def test_interpolate_hdiv_hcurl_exact(fam, deg, field):
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, fam, deg)
    x, y = SpatialCoordinate(m)
    u = Function(V).interpolate(field(x, y))
    assert errornorm(field(x, y), u) < 1e-12


def test_interpolate_cr_and_dg0_and_real():
    m = UnitSquareMesh(3, 3)
    x, y = SpatialCoordinate(m)
    cr = Function(FunctionSpace(m, "CR", 1)).interpolate(1 + x - y)
    assert cr.at((0.3, 0.4)) == pytest.approx(0.9)
    dg0 = Function(FunctionSpace(m, "DG", 0)).assign(2.0)
    assert assemble(dg0 * dx) == pytest.approx(2.0)
    r = Function(FunctionSpace(m, "R", 0)).assign(3.0)
    assert assemble(r * dx) == pytest.approx(3.0)


def test_hdiv_dirichlet_normal_bc_and_div_free_projection():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "RT", 1)
    x, y = SpatialCoordinate(m)
    u = Function(V)
    DirichletBC(V, as_vector((1.0, 0.0)), (1, 2)).apply(u)
    n = FacetNormal(m)
    assert assemble(dot(u, n) * ds(2)) == pytest.approx(1.0)
    assert assemble(dot(u, n) * ds(1)) == pytest.approx(-1.0)
    assert assemble(dot(u, n) * ds(3)) == pytest.approx(0.0, abs=1e-14)


def test_hcurl_maxwell_like_solve_exact_for_n1curl():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "N1curl", 1)
    x, y = SpatialCoordinate(m)
    ex = as_vector((1 - y, 2 + x))
    u, v = TrialFunction(V), TestFunction(V)
    a = (curl(u) * curl(v) + inner(u, v)) * dx
    L = (curl(ex) * curl(v) + inner(ex, v)) * dx
    uh = Function(V)
    solve(a == L, uh)
    assert errornorm(ex, uh) < 1e-10


def test_nullspace_with_explicit_function_basis():
    m = UnitSquareMesh(8, 8)
    V = FunctionSpace(m, "CG", 1)
    x, y = SpatialCoordinate(m)
    u, v = TrialFunction(V), TestFunction(V)
    one = Function(V).assign(1.0)
    ns = VectorSpaceBasis([one])
    ns.orthonormalize()
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == pi**2 * cos(pi * x) * v * dx, uh, nullspace=ns)
    mean = assemble(uh * dx)
    assert math.sqrt(assemble((uh - mean - cos(pi * x)) ** 2 * dx)) < 2e-2


def test_free_elasticity_with_rigid_body_nullspace():
    m = UnitSquareMesh(6, 6)
    V = VectorFunctionSpace(m, "CG", 1)
    x, y = SpatialCoordinate(m)
    E, nu = 1.0, 0.3
    mu, lmbda = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    def sigma(w):
        return 2 * mu * sym(grad(w)) + lmbda * tr(sym(grad(w))) * Identity(2)

    u, v = TrialFunction(V), TestFunction(V)
    a = inner(sigma(u), sym(grad(v))) * dx
    L = inner(Constant((1.0, 0.0)), v) * ds(1) + inner(Constant((-1.0, 0.0)), v) * ds(2)
    modes = [Function(V).interpolate(as_vector(c)) for c in ((1.0, 0.0), (0.0, 1.0), (-y, x))]
    ns = VectorSpaceBasis(modes)
    ns.orthonormalize()
    uh = Function(V)
    solve(a == L, uh, nullspace=ns)
    stretch = assemble(uh[0] * ds(2)) - assemble(uh[0] * ds(1))
    assert stretch < -0.5 * 0.5
    r = assemble(inner(sigma(uh), sym(grad(v))) * dx - L)
    free = np.asarray(r.dat.data_ro)
    assert np.abs(free).max() < 1e-10

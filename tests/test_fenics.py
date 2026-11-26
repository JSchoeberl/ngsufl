"""Tests of the legacy FEniCS front end ``ngsufl.fenics``."""

import math

import ngsolve as ngs
import numpy as np
import pytest
import ufl

from ngsufl.fenics import *
from ngsufl.fenics import ConvergenceError, _convert, _linear_options

isc = np.isclose


def vals(u):
    return u.vector().get_local()


# ---------------------------------------------------------------- meshes

def test_unit_square_mesh_counts_and_extent():
    m = UnitSquareMesh(4, 3)
    assert m.num_cells() == 2 * 4 * 3 and m.num_vertices() == 5 * 4
    X = m.coordinates()
    assert X.shape == (20, 2)
    assert np.allclose(X.min(axis=0), 0) and np.allclose(X.max(axis=0), 1)
    assert m.cells().shape == (24, 3) and m.cells().max() == 19
    assert m.geometry().dim() == 2 and m.topology().dim() == 2
    assert isc(m.hmax(), math.hypot(0.25, 1 / 3)) and isc(m.hmin(), math.hypot(0.25, 1 / 3))
    assert str(m.ufl_cell()) == "triangle"


@pytest.mark.parametrize("diagonal,sign", [("right", 1), ("left", -1), (None, 1)])
def test_diagonal_direction(diagonal, sign):
    m = UnitSquareMesh(3, 3) if diagonal is None else UnitSquareMesh(3, 3, diagonal)
    X, C = m.coordinates(), m.cells()
    for c in C:
        P = X[c]
        pairs = [(a, b) for a in range(3) for b in range(a + 1, 3)]
        a, b = max(pairs, key=lambda ab: np.linalg.norm(P[ab[0]] - P[ab[1]]))
        d = P[a] - P[b]
        assert d[0] * d[1] * sign > 0


def test_diagonal_not_supported():
    for d in ("crossed", "left/right"):
        with pytest.raises(NotImplementedError):
            UnitSquareMesh(2, 2, d)


def test_rectangle_mesh():
    m = RectangleMesh(Point(-1, 2), Point(3, 4), 8, 2)
    X = m.coordinates()
    assert np.allclose(X.min(axis=0), (-1, 2)) and np.allclose(X.max(axis=0), (3, 4))
    assert m.num_cells() == 32
    assert isc(assemble(Constant(1.0) * dx(domain=m)), 8.0)


def test_interval_meshes():
    m = UnitIntervalMesh(5)
    assert m.num_cells() == 5 and m.coordinates().shape == (6, 1) and m.geometry().dim() == 1
    m = IntervalMesh(4, -1.0, 3.0)
    X = m.coordinates()
    assert isc(X.min(), -1) and isc(X.max(), 3) and isc(m.hmax(), 1.0)


def test_box_and_cube_mesh():
    m = UnitCubeMesh(2, 2, 2)
    assert m.num_cells() == 6 * 8 and m.num_vertices() == 27 and m.geometry().dim() == 3
    assert m.coordinates().shape == (27, 3)
    b = BoxMesh(Point(0, 0, 0), Point(2, 1, 0.5), 4, 2, 1)
    X = b.coordinates()
    assert np.allclose(X.max(axis=0), (2, 1, 0.5)) and np.allclose(X.min(axis=0), 0)
    assert isc(assemble(Constant(1.0) * dx(domain=b)), 1.0)


def test_mesh_is_core_mesh_with_ngsolve_mesh():
    m = UnitSquareMesh(2, 2)
    assert isinstance(m.ngs, ngs.Mesh) and m.geometric_dimension() == 2 and m().num_cells() == 8


def test_point():
    p = Point(1.0, 2.0)
    assert (p.x(), p.y(), p.z()) == (1.0, 2.0, 0.0)
    assert p[0] == 1.0 and p[1] == 2.0 and len(p) == 3
    assert np.allclose(p.array(), (1, 2, 0))
    assert isc(p.distance(Point(4.0, 6.0)), 5.0) and isc(Point(3, 4).norm(), 5.0)
    assert Point((1.0, 2.0, 3.0)).z() == 3.0


# ---------------------------------------------------------------- near, between

def test_near_between():
    assert near(0.5, 0.5) and near(0.5 + 1e-16, 0.5) and not near(0.51, 0.5)
    assert near(0.51, 0.5, 0.02)
    assert between(0.5, (0.0, 1.0)) and between(1.0 + 1e-16, (0.0, 1.0)) and not between(1.1, (0.0, 1.0))
    assert DOLFIN_EPS == 3e-16 and isc(DOLFIN_PI, math.pi)


# ---------------------------------------------------------------- Expression parser

PT = (0.3, 0.7)
X0, X1 = PT


def cases():
    return [
        ("1 + 2*3", 7.0), ("2 + 3*4 - 1", 13.0), ("-2*3", -6.0), ("2*-3", -6.0), ("10/4", 2.5),
        ("8/4/2", 1.0), ("1 - 2 - 3", -4.0), ("(1+2)*3", 9.0), ("2.", 2.0), ("1e-3 + .5", 0.501),
        ("2.5e1", 25.0), ("1E2", 100.0), ("+x[1]", X1), ("-x[0] + 2", 2 - X0),
        ("-(x[0] + x[1])", -(X0 + X1)), ("x[0]*x[1]", X0 * X1), ("x[1]/x[0]", X1 / X0),
        ("pow(x[0], 2)", X0 ** 2), ("pow(x[1], 0.5)", X1 ** 0.5), ("sqrt(x[1])", math.sqrt(X1)),
        ("exp(-x[0])", math.exp(-X0)), ("log(x[1])", math.log(X1)), ("sin(x[0])", math.sin(X0)),
        ("cos(x[0])", math.cos(X0)), ("tan(x[0])", math.tan(X0)), ("asin(x[0])", math.asin(X0)),
        ("acos(x[0])", math.acos(X0)), ("atan(x[1])", math.atan(X1)),
        ("atan2(x[1], x[0])", math.atan2(X1, X0)), ("sinh(x[0])", math.sinh(X0)),
        ("cosh(x[0])", math.cosh(X0)), ("tanh(x[0])", math.tanh(X0)), ("fabs(-x[0])", X0),
        ("abs(x[0] - x[1])", abs(X0 - X1)), ("min(x[0], x[1])", min(X0, X1)),
        ("max(x[0], x[1])", max(X0, X1)), ("fmin(x[0], x[1])", min(X0, X1)),
        ("fmax(x[0], x[1])", max(X0, X1)), ("pi", math.pi), ("DOLFIN_PI", math.pi),
        ("M_PI", math.pi), ("DOLFIN_EPS", 3e-16), ("sin(pi*x[0])*cos(2*pi*x[1])",
                                                    math.sin(math.pi * X0) * math.cos(2 * math.pi * X1)),
        ("x[0] < x[1] ? 1 : 2", 1.0), ("x[0] > x[1] ? 1 : 2", 2.0), ("x[0] <= 0.3 ? 1 : 0", 1.0),
        ("x[0] >= 0.3 ? 1 : 0", 1.0), ("x[0] == 0.3 ? 1 : 0", 1.0), ("x[0] != 0.3 ? 1 : 0", 0.0),
        ("x[0] < 0.3 ? 1 : 0", 0.0), ("x[0] > 0.3 ? 1 : 0", 0.0),
        ("x[0] > 0.5 ? 1 : x[1] > 0.5 ? 2 : 3", 2.0),
        ("x[0] < 0.5 && x[1] > 0.5 ? 1 : 0", 1.0), ("x[0] > 0.5 || x[1] > 0.5 ? 1 : 0", 1.0),
        ("x[0] > 0.5 || x[1] > 0.9 ? 1 : 0", 0.0), ("x[0] < 0.5 && x[1] > 0.9 ? 1 : 0", 0.0),
        ("1 || 0 && 0", 1.0), ("(1 || 0) && 0", 0.0), ("!(x[0] > 0.5)", 1.0),
        ("!(x[0] < 0.5)", 0.0), ("!0 + !1", 1.0), ("(x[0] < 0.5) + (x[1] > 0.5)", 2.0),
        ("2 * (x[0] < 0.5)", 2.0), ("near(x[0], 0.3)", 1.0), ("near(x[0], 0.5)", 0.0),
        ("near(x[0], 0.31, 0.02)", 1.0), ("near(x[0], 0.5) ? 5 : 6", 6.0),
        ("x[0] + 1 < 2 ? x[1] : 0", X1), ("1 < 2 == 1 ? 7 : 8", 7.0),
        ("  1 +\t2 ", 3.0),
    ]


@pytest.fixture(scope="module")
def sq():
    return UnitSquareMesh(4, 4)


@pytest.mark.parametrize("code,expected", cases())
def test_expression_scalar_both_evaluators(sq, code, expected):
    e = Expression(code, degree=2, mesh=sq)
    assert isc(e(PT), expected, rtol=1e-12, atol=1e-14)
    assert isc(e(*PT), expected, rtol=1e-12, atol=1e-14)
    cf = to_cf(e)
    assert isc(cf(sq.ngs(*PT)), expected, rtol=1e-12, atol=1e-14)


def test_expression_integral_matches_python_value(sq):
    e = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2, mesh=sq)
    assert isc(assemble(e * dx), 1 + 1 / 3 + 2 / 3)


def test_expression_vector_and_tensor(sq):
    e = Expression(("x[1]", "-x[0]"), degree=1, mesh=sq)
    assert e.ufl_shape == (2,) and np.allclose(e(PT), (X1, -X0))
    assert np.allclose(to_cf(e)(sq.ngs(*PT)), (X1, -X0))
    assert isc(assemble(inner(e, e) * dx), 2 / 3)
    t = Expression((("1", "x[0]"), ("x[1]", "2")), degree=1, mesh=sq)
    assert t.ufl_shape == (2, 2) and np.allclose(t(PT), [[1, X0], [X1, 2]])
    assert np.allclose(np.array(to_cf(t)(sq.ngs(*PT))).reshape(2, 2), [[1, X0], [X1, 2]])
    c = Expression(("0", 1.5), degree=0, mesh=sq)
    assert np.allclose(c(PT), (0, 1.5))
    assert Expression(["1", "2", "3"], degree=0, mesh=sq).ufl_shape == (3,)


def test_expression_parameters_update(sq):
    g = Expression("exp(-a*pow(x[0], 2)) + t", degree=2, a=5, t=0, mesh=sq)
    assert g.a == 5 and g.t == 0
    v0 = g(0.3, 0.1)
    assert isc(v0, math.exp(-5 * 0.09))
    I0 = assemble(g * dx)
    g.t = 0.3
    assert g.t == 0.3 and isc(g(0.3, 0.1), v0 + 0.3)
    assert isc(assemble(g * dx), I0 + 0.3)
    assert isc(to_cf(g)(sq.ngs(0.3, 0.1)), v0 + 0.3)
    g.a = 2.0
    assert isc(g(0.3, 0.1), math.exp(-2 * 0.09) + 0.3) and isc(to_cf(g)(sq.ngs(0.3, 0.1)), g(0.3, 0.1))
    g.t = 1
    assert isinstance(g.t, float) and g.t == 1.0


def test_expression_parameter_conflicting_with_ufl_attributes(sq):
    e = Expression("T*x[0] + dx", degree=1, T=2.0, dx=1.0, mesh=sq)
    assert e.T == 2.0 and e.dx == 1.0 and isc(e((0.5, 0.5)), 2.0)
    e.T = 4.0
    assert isc(e((0.5, 0.5)), 3.0)


def test_expression_vector_parameters_update(sq):
    e = Expression(("a*x[0]", "b"), degree=1, a=1.0, b=2.0, mesh=sq)
    e.a = 3.0
    assert np.allclose(e((0.5, 0.5)), (1.5, 2.0)) and np.allclose(to_cf(e)(sq.ngs(0.5, 0.5)), (1.5, 2.0))


def test_expression_element_keyword(sq):
    P2 = FiniteElement("P", triangle, 2)
    e = Expression("x[0]*x[0]", element=P2, mesh=sq)
    assert e.ufl_element().embedded_superdegree == 2
    ev = Expression(("x[0]", "x[1]"), element=VectorElement("P", triangle, 1), mesh=sq)
    assert ev.ufl_shape == (2,)
    assert Expression("1", 3, mesh=sq).ufl_element().embedded_superdegree == 3


def test_expression_degree_reaches_quadrature(sq):
    e1 = Expression("x[0]", degree=1, mesh=sq)
    e5 = Expression("x[0]", degree=5, mesh=sq)
    V = FunctionSpace(sq, "P", 1)
    v = TestFunction(V)
    from ufl.algorithms import estimate_total_polynomial_degree as deg
    assert deg(e1 * v) == 2
    assert deg(e5 * v) == 6


def test_expression_3d_and_dimension_check():
    m3 = UnitCubeMesh(2, 2, 2)
    e = Expression("x[0] + 2*x[1] + 3*x[2]", degree=1, mesh=m3)
    assert isc(e((0.1, 0.2, 0.3)), 1.4) and isc(assemble(e * dx), 3.0)
    m2 = UnitSquareMesh(2, 2)
    with pytest.raises(ValueError, match="dimension"):
        Expression("x[2]", degree=1, mesh=m2)


@pytest.mark.parametrize("bad", [
    "1 +", "x[0] +* 2", "(1+2", "1+2)", "sin(1, 2)", "foo(1)", "x[a]", "x[1.5]", "1 $ 2", "",
    "   ", "x[0] x[1]", "pow(1)", "x[0", "2 x[0]", "1 ? 2", "? 1 : 2", "x[0] ^ 2", "a*x[0]",
    "on_boundary", "1 ? 2 :", "sin()", "near(1)", "x[0]**2", "x[0] && ", "[1]"])
def test_expression_syntax_errors(sq, bad):
    with pytest.raises(ValueError):
        Expression(bad, degree=1, mesh=sq)


def test_expression_requires_degree(sq):
    with pytest.raises(ValueError, match="degree"):
        Expression("x[0]", mesh=sq)
    with pytest.raises(TypeError):
        Expression("a", degree=1, a="text", mesh=sq)
    with pytest.raises(NotImplementedError):
        Expression(degree=1, mesh=sq)


def test_expression_where_ufl_works(sq):
    e = Expression("x[0]*x[1]", degree=2, mesh=sq)
    V = FunctionSpace(sq, "P", 2)
    # interpolate
    w = interpolate(e, V)
    assert np.allclose(w.compute_vertex_values(), e.compute_vertex_values(sq))
    # project
    p = project(e, V)
    assert errornorm(e, p) < 1e-13
    # forms
    v = TestFunction(FunctionSpace(sq, "P", 1))
    assert isc(assemble(e * v * dx).get_local().sum(), assemble(e * dx))
    # errornorm
    assert errornorm(e, w) < 1e-13
    # BC
    bc = DirichletBC(V, e, "on_boundary")
    g = Function(V)
    bc.apply(g.vector())
    vals_bc = bc.get_boundary_values()
    assert len(vals_bc) == 32
    # point evaluation of a Function built from it
    assert isc(w(0.3, 0.7), 0.21)
    # ufl algebra
    assert isc(assemble(sin(e) * dx + e ** 2 * dx), assemble(sin(e) * dx) + assemble(e * e * dx))


def test_expression_compute_vertex_values_vector_ordering(sq):
    e = Expression(("x[0]", "2*x[1]"), degree=1, mesh=sq)
    v = e.compute_vertex_values(sq)
    X = sq.coordinates()
    n = len(X)
    assert np.allclose(v[:n], X[:, 0]) and np.allclose(v[n:], 2 * X[:, 1])


def test_expression_eval_and_misc(sq):
    e = Expression(("x[0]", "x[1]"), degree=1, mesh=sq)
    out = np.zeros(2)
    e.eval(out, np.array([0.2, 0.4]))
    assert np.allclose(out, (0.2, 0.4))
    assert e.value_rank() == 1 and e.value_dimension(0) == 2 and e.geometric_dimension() == 2


def test_expression_follows_latest_mesh_in_forms():
    e = Expression("x[0]", degree=1)
    for n in (2, 4):
        m = UnitSquareMesh(n, n)
        V = FunctionSpace(m, "P", 1)
        assert isc(assemble(e * TestFunction(V) * dx).get_local().sum(), 0.5)
        assert isc(assemble(e * dx), 0.5)
        assert errornorm(e, interpolate(e, V)) < 1e-14


# ---------------------------------------------------------------- sub domains and bcs

def _left_dofs(V):
    return list(V.fes.GetDofs(V.mesh.ngs.Boundaries("left")))


def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)


SUBDOMAINS = {
    "function": lambda: boundary_left,
    "string": lambda: "on_boundary && near(x[0], 0)",
    "string_eps": lambda: "x[0] < DOLFIN_EPS",
    "string_between": lambda: "on_boundary && x[0] < 1e-10 && x[0] > -1e-10",
    "subdomain": lambda: Left(),
    "compiled": lambda: CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0),
    "compiled_plain": lambda: CompiledSubDomain("x[0] < DOLFIN_EPS"),
    "auto": lambda: AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[0], 0)),
    "lambda": lambda: (lambda x, on_boundary: abs(x[0]) < 1e-12 and on_boundary),
}


@pytest.mark.parametrize("kind", list(SUBDOMAINS))
@pytest.mark.parametrize("deg", [1, 2])
def test_dirichlet_subdomain_forms_give_same_dofs(kind, deg):
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "P", deg)
    bc = DirichletBC(V, Constant(2.0), SUBDOMAINS[kind]())
    assert list(bc.dofs()) == _left_dofs(V)
    assert sum(bc.dofs()) == deg * 4 + 1


def test_dirichlet_on_boundary_and_values():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "P", 2)
    bc = DirichletBC(V, Constant(1.0), "on_boundary")
    assert sum(bc.dofs()) == 4 * 4 * 2
    bc2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary: on_boundary)
    assert list(bc.dofs()) == list(bc2.dofs())
    x = Function(V)
    bc.apply(x.vector())
    assert isc(x.vector().max(), 1.0) and isc(x.vector().min(), 0.0)


def test_dirichlet_compiled_subdomain_parameter_update():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "P", 1)
    sd = CompiledSubDomain("near(x[0], side)", side=0.0)
    assert sum(DirichletBC(V, 0.0, sd).dofs()) == 5
    sd.side = 1.0
    assert sd.side == 1.0 and sd.inside(np.array([1.0, 0.3]), True) and not sd.inside(np.array([0.0, 0.3]), True)
    bc = DirichletBC(V, 5.0, sd)
    assert sum(bc.dofs()) == 5
    w = Function(V)
    bc.apply(w.vector())
    assert np.allclose(np.sort(w.compute_vertex_values())[-5:], 5.0)


def test_dirichlet_subdomain_methods_and_errors():
    sd = SubDomain()
    with pytest.raises(NotImplementedError, match="mark"):
        sd.mark(None, 1)
    with pytest.raises(NotImplementedError):
        Left().map(None, None)
    with pytest.raises(NotImplementedError):
        SubDomain().inside(np.zeros(2), True)
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "P", 1)
    with pytest.raises(ValueError):
        DirichletBC(V, 0.0, "on_boundary && nonsense(")
    with pytest.raises(ValueError):
        DirichletBC(V, 0.0, "on_boundary", method="bogus")
    with pytest.raises(ValueError):
        CompiledSubDomain("near(x[0], side)")


@pytest.mark.parametrize("method", ["topological", "geometric"])
def test_dirichlet_method_keyword(method):
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "P", 1)
    bc = DirichletBC(V, 0.0, "near(x[0], 0)", method)
    assert sum(bc.dofs()) == 5


def test_dirichlet_pointwise():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "P", 2)
    bc = DirichletBC(V, Constant(3.0), "near(x[0], 0.5) && near(x[1], 0.5)", method="pointwise")
    assert sum(bc.dofs()) == 1
    assert list(bc.get_boundary_values().values()) == [3.0]
    # a pure Neumann problem becomes well posed by pinning the solution at one point
    m = UnitSquareMesh(16, 16)
    V = FunctionSpace(m, "P", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    ue = cos(math.pi * x[0]) * cos(math.pi * x[1])
    f = 2 * math.pi ** 2 * ue
    bcp = DirichletBC(V, Constant(1.0), lambda x, ob: near(x[0], 0.0) and near(x[1], 0.0), method="pointwise")
    uh = Function(V)
    solve(dot(grad(u), grad(v)) * dx == f * v * dx, uh, bcp)
    ue_h = interpolate(ue, V)
    d = uh.compute_vertex_values() - ue_h.compute_vertex_values()
    assert np.ptp(d) < 2e-2 and isc(uh(0.0, 0.0), 1.0)


def test_dirichlet_value_and_homogenize():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "P", 1)
    bc = DirichletBC(V, 4.0, "on_boundary")
    assert set(bc.get_boundary_values().values()) == {4.0}
    bc.homogenize()
    assert set(bc.get_boundary_values().values()) == {0.0}
    bc.set_value(2.0)
    assert set(bc.get_boundary_values().values()) == {2.0}
    assert bc.function_space() is V


def test_dirichlet_nodal_interpolation_of_data():
    m = UnitSquareMesh(4, 4)
    for deg in (1, 2, 3):
        V = FunctionSpace(m, "P", deg)
        g = Expression("sin(3*x[0] + x[1])", degree=deg + 2)
        w = Function(V)
        DirichletBC(V, g, "on_boundary").apply(w.vector())
        full = interpolate(g, V)
        bcdofs = np.array(list(DirichletBC(V, g, "on_boundary").dofs()), dtype=bool)
        assert np.allclose(vals(w)[bcdofs], vals(full)[bcdofs])
        X = m.coordinates()
        assert np.allclose(full.compute_vertex_values(), np.sin(3 * X[:, 0] + X[:, 1]))


# ---------------------------------------------------------------- spaces and functions

def test_function_space_constructors():
    m = UnitSquareMesh(3, 3)
    assert FunctionSpace(m, "P", 1).dim() == 16
    assert FunctionSpace(m, "Lagrange", 2).dim() == 49
    assert FunctionSpace(m, "CG", 1).dim() == 16
    assert FunctionSpace(m, "DG", 0).dim() == 18
    assert FunctionSpace(m, "DG", 1).dim() == 54
    assert VectorFunctionSpace(m, "P", 2).dim() == 98
    assert VectorFunctionSpace(m, "P", 1, dim=2).dim() == 32
    assert TensorFunctionSpace(m, "P", 1).dim() == 64
    assert FunctionSpace(m, FiniteElement("P", triangle, 2)).dim() == 49
    assert FunctionSpace(m, FiniteElement("Lagrange", m.ufl_cell(), 1)).dim() == 16
    assert FunctionSpace(m, VectorElement("P", triangle, 1)).dim() == 32
    assert FunctionSpace(m, "N1curl", 1).dim() == 33
    with pytest.raises(NotImplementedError, match="constrained_domain"):
        FunctionSpace(m, "P", 1, constrained_domain=object())
    assert FunctionSpace(m, "P", 1).mesh() is m


def test_mixed_spaces_and_split():
    m = UnitSquareMesh(3, 3)
    P2 = VectorElement("P", triangle, 2)
    P1 = FiniteElement("P", triangle, 1)
    for W in (FunctionSpace(m, MixedElement([P2, P1])), FunctionSpace(m, P2 * P1)):
        assert W.dim() == 98 + 16 and W.num_sub_spaces() == 2
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)
        assert u.ufl_shape == (2,) and p.ufl_shape == ()
        assert TrialFunction(W).geometric_dimension() == 2
        U, P = split(TrialFunction(W))
        assert U.ufl_shape == (2,)
        w = Function(W)
        assert len(w.split()) == 2
        assert w.sub(0).function_space().dim() == 98
    assert W.sub(0).collapse().dim() == 98 and W.sub(1).collapse().dim() == 16


def test_function_split_deepcopy_and_shallow():
    m = UnitSquareMesh(3, 3)
    W = FunctionSpace(m, VectorElement("P", triangle, 2) * FiniteElement("P", triangle, 1))
    w = Function(W)
    w.interpolate(Expression(("1", "2", "3"), degree=1))
    u, p = w.split()
    assert np.allclose(p.compute_vertex_values(), 3)
    assert np.allclose(u.compute_vertex_values(), [1] * 16 + [2] * 16)
    u2, p2 = w.split(deepcopy=True)
    w.vector()[:] = 0.0
    assert np.allclose(u.compute_vertex_values(), 0)
    assert np.allclose(u2.compute_vertex_values(), [1] * 16 + [2] * 16)
    assert np.allclose(p2.compute_vertex_values(), 3)
    assert isinstance(u2, Function) and u2.function_space().dim() == 98


def test_function_basics():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "P", 1)
    u = Function(V, name="temperature")
    assert u.name() == "temperature" and u.geometric_dimension() == 2
    u.rename("T", "label")
    assert u.name() == "T" and u.function_space() is V
    u.interpolate(Expression("x[0] + 2*x[1]", degree=1))
    assert isc(u(0.25, 0.5), 1.25) and isc(u((0.25, 0.5)), 1.25)
    w = u.copy()
    w.vector()[:] = 2 * w.vector().get_local()
    assert isc(w(0.25, 0.5), 2.5) and isc(u(0.25, 0.5), 1.25)
    u.assign(w)
    assert isc(u(0.25, 0.5), 2.5)
    assert u.vector().size() == 25 and isc(u.vector().max(), 6.0) and isc(u.vector().min(), 0.0)
    assert isc(u.vector().norm("linf"), 6.0)
    uv = VectorFunctionSpace(m, "P", 1)
    f = Function(uv).interpolate(Expression(("x[1]", "-x[0]"), degree=1))
    assert np.allclose(f(0.25, 0.5), (0.5, -0.25))
    assert f.compute_vertex_values().shape == (50,)


def test_interpolate_project_return_legacy_functions():
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, "P", 1)
    for f in (interpolate(Constant(1.0), V), project(Constant(1.0), V)):
        assert isinstance(f, Function) and np.allclose(f.compute_vertex_values(), 1.0)
        f.split  # legacy API present
    out = Function(V)
    r = project(Expression("x[0]", degree=1), V, function=out)
    assert r is out and isc(out(0.5, 0.5), 0.5)
    assert isc(project(interpolate(Expression("x[0]", degree=1), V)).vector().max(), 1.0)
    with pytest.raises(ValueError):
        project(Expression("x[0]", degree=1))


def test_project_reproduces_polynomials_and_differs_from_interpolation():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "P", 2)
    g = Expression("x[0]*x[0] + x[1]", degree=2)
    assert errornorm(g, project(g, V)) < 1e-14
    V1 = FunctionSpace(m, "P", 1)
    assert errornorm(g, project(g, V1)) < errornorm(g, interpolate(g, V1))


def test_trial_and_test_function_equal_ufl_arguments():
    V = FunctionSpace(UnitSquareMesh(2, 2), "P", 1)
    u, v = TrialFunction(V), TestFunction(V)
    assert u == ufl.Argument(V, 1) and ufl.Argument(V, 0) == v and u != v
    assert hash(u) == hash(ufl.Argument(V, 1))
    assert u.number() == 1 and v.number() == 0 and isinstance(u, ufl.Argument)


def test_geometric_dimension_on_arguments_and_forms_machinery():
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, "P", 1)
    u, v = TrialFunction(V), TestFunction(V)
    assert u.geometric_dimension() == v.geometric_dimension() == 2
    uh = Function(V)
    F = (1 + uh ** 2) * dot(grad(uh), grad(v)) * dx - Constant(1.0) * v * dx
    J = derivative(F, uh, u)
    assert [a.number() for a in J.arguments()] == [0, 1]
    J2 = derivative(F, uh)
    assert len(J2.arguments()) == 2
    a = u * v * dx + dot(grad(u), grad(v)) * dx - Constant(1.0) * v * dx
    assert len(lhs(a).arguments()) == 2 and len(rhs(a).arguments()) == 1
    A, L = system(a)
    assert len(A.arguments()) == 2 and len(L.arguments()) == 1
    assert len(action(lhs(a), uh).arguments()) == 1
    assert len(adjoint(lhs(a)).arguments()) == 2
    assert len(replace(L, {v: TestFunction(V)}).arguments()) == 1
    # mixed-in plain UFL arguments
    form = u * ufl.Argument(V, 0) * dx
    assert isc(assemble(form).array().sum(), 1.0)
    mixed = ufl.Argument(V, 1) * v * dx + u * v * dx
    assert isc(assemble(mixed).array().sum(), 2.0)


def test_derivative_with_legacy_arguments_matches_native_newton_matrix():
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, "P", 1)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V).interpolate(Expression("1 + x[0]", degree=1))
    F = uh ** 2 * v * dx
    J = derivative(F, uh, u)
    A = assemble(J).array()
    B = assemble(2 * uh * u * v * dx).array()
    assert np.allclose(A, B)


# ---------------------------------------------------------------- assemble, solve, norms

def poisson_setup(n=6, deg=1):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "P", deg)
    u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
    bc = DirichletBC(V, u_D, lambda x, on_boundary: on_boundary)
    u, v = TrialFunction(V), TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    L = Constant(-6.0) * v * dx
    return mesh, V, u_D, bc, a, L


def test_classic_poisson_program():
    mesh, V, u_D, bc, a, L = poisson_setup(8)
    u = Function(V)
    solve(a == L, u, bc)
    assert isc(errornorm(u_D, u, "L2"), 0.008235098073354905, rtol=1e-8)
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    assert np.abs(vertex_values_u_D - vertex_values_u).max() < 1e-12
    X = mesh.coordinates()
    assert np.abs(vertex_values_u - (1 + X[:, 0] ** 2 + 2 * X[:, 1] ** 2)).max() < 1e-12
    assert isc(norm(u, "L2"), math.sqrt(assemble(u * u * dx)))
    assert errornorm(u_D, u, "H1") > errornorm(u_D, u, "L2")
    assert errornorm(u_D, u, norm_type="L2", degree_rise=1) > 0
    assert isc(errornorm(u_D, u, "L2", 3, mesh), errornorm(u_D, u))


def test_poisson_matches_native_ngsolve():
    mesh, V, u_D, bc, a, L = poisson_setup(5)
    u = Function(V)
    solve(a == L, u, bc)
    fes = ngs.H1(mesh.ngs, order=1, dirichlet=".*")
    gf = ngs.GridFunction(fes)
    gf.Interpolate(1 + ngs.x ** 2 + 2 * ngs.y ** 2, definedon=mesh.ngs.Boundaries(".*"))
    uu, vv = fes.TnT()
    A = ngs.BilinearForm(ngs.grad(uu) * ngs.grad(vv) * ngs.dx).Assemble()
    f = ngs.LinearForm(-6.0 * vv * ngs.dx).Assemble()
    gf.vec.data += A.mat.Inverse(fes.FreeDofs()) * (f.vec - A.mat * gf.vec)
    pts = mesh.coordinates()
    nat = np.array([gf(mesh.ngs(*p)) for p in pts])
    assert np.allclose(u.compute_vertex_values(), nat, atol=1e-12)


def test_poisson_p2_exact():
    mesh, V, u_D, bc, a, L = poisson_setup(4, 2)
    u = Function(V)
    solve(a == L, u, bc)
    assert errornorm(u_D, u) < 1e-13


def test_solve_bcs_forms():
    mesh, V, u_D, bc, a, L = poisson_setup(4)
    ref = Function(V)
    solve(a == L, ref, bc)
    for bcs in ([bc], (bc,)):
        u = Function(V)
        solve(a == L, u, bcs)
        assert np.allclose(vals(u), vals(ref))
    u = Function(V)
    solve(a == L, u, bcs=bc)
    assert np.allclose(vals(u), vals(ref))


def test_heat_equation_time_dependent_expression():
    alpha, beta = 3.0, 1.2
    nx = ny = 8
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, "P", 1)
    u_D = Expression("1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t", degree=2, alpha=alpha, beta=beta, t=0)
    bc = DirichletBC(V, u_D, lambda x, on_boundary: on_boundary)
    u_n = interpolate(u_D, V)
    T, num_steps = 2.0, 10
    dt = T / num_steps
    u, v = TrialFunction(V), TestFunction(V)
    f = Constant(beta - 2 - 2 * alpha)
    F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx
    a, L = lhs(F), rhs(F)
    u = Function(V)
    t = 0
    for n in range(num_steps):
        t += dt
        u_D.t = t
        solve(a == L, u, bc)
        error = np.abs(u_D.compute_vertex_values(mesh) - u.compute_vertex_values(mesh)).max()
        assert error < 1e-12
        u_n.assign(u)
    assert isc(u_D.t, T)


def test_heat_equation_assembled_matrix_route_equals_variational_route():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "P", 1)
    u_D = Expression("1 + x[0]*x[0] + 3*x[1]*x[1] + 1.2*t", degree=2, t=0.0)
    bc = DirichletBC(V, u_D, "on_boundary")
    dt = 0.2
    u, v = TrialFunction(V), TestFunction(V)
    u_n1 = interpolate(u_D, V)
    u_n2 = interpolate(u_D, V)
    f = Constant(1.2 - 2 - 6)
    a = u * v * dx + dt * dot(grad(u), grad(v)) * dx
    A = assemble(a)
    bc.apply(A)
    solver = LUSolver(A)
    sol1, sol2 = Function(V), Function(V)
    t = 0.0
    for n in range(5):
        t += dt
        u_D.t = t
        L1 = (u_n1 + dt * f) * v * dx
        b = assemble(L1)
        bc.apply(b)
        solver.solve(sol1.vector(), b)
        solve(a == (u_n2 + dt * f) * v * dx, sol2, bc)
        assert np.allclose(vals(sol1), vals(sol2), atol=1e-11)
        u_n1.assign(sol1)
        u_n2.assign(sol2)
    assert np.abs(u_D.compute_vertex_values() - sol2.compute_vertex_values()).max() < 1e-12


def q(u):
    return 1 + u ** 2


@pytest.mark.parametrize("deg", [1, 2])
def test_nonlinear_poisson_manufactured(deg):
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "P", deg)
    x = SpatialCoordinate(mesh)
    u_ufl = 1 + x[0] + 2 * x[1]
    u_D = Expression("1 + x[0] + 2*x[1]", degree=1)
    bc = DirichletBC(V, u_D, "on_boundary")
    f = -div(q(u_ufl) * grad(u_ufl))
    u = Function(V)
    v = TestFunction(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    solve(F == 0, u, bc)
    assert errornorm(u_D, u) < 1e-9
    X = mesh.coordinates()
    assert np.abs(u.compute_vertex_values() - (1 + X[:, 0] + 2 * X[:, 1])).max() < 1e-9


def test_nonlinear_poisson_with_explicit_jacobian_and_parameters():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "P", 1)
    x = SpatialCoordinate(mesh)
    u_ufl = 1 + x[0] + 2 * x[1]
    u_D = Expression("1 + x[0] + 2*x[1]", degree=1)
    bc = DirichletBC(V, u_D, "on_boundary")
    f = -div(q(u_ufl) * grad(u_ufl))
    u = Function(V)
    v = TestFunction(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    J = derivative(F, u, TrialFunction(V))
    prm = {"newton_solver": {"relative_tolerance": 1e-8, "absolute_tolerance": 1e-9,
                             "maximum_iterations": 20, "relaxation_parameter": 1.0}}
    solve(F == 0, u, bc, J=J, solver_parameters=prm)
    assert errornorm(u_D, u) < 1e-8
    u2 = Function(V)
    F2 = q(u2) * dot(grad(u2), grad(v)) * dx - f * v * dx
    solve(F2 == 0, u2, bc, solver_parameters={"newton_solver": {"linear_solver": "gmres",
                                                                "preconditioner": "none",
                                                                "relative_tolerance": 1e-9}})
    assert np.allclose(vals(u), vals(u2), atol=1e-7)


def nonlinear_problem(damping=None):
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "P", 1)
    x = SpatialCoordinate(mesh)
    u_ufl = 1 + x[0] + 2 * x[1]
    bc = DirichletBC(V, Expression("1 + x[0] + 2*x[1]", degree=1), "on_boundary")
    f = -div(q(u_ufl) * grad(u_ufl))
    u = Function(V)
    v = TestFunction(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    J = derivative(F, u, TrialFunction(V))
    return V, u, NonlinearVariationalProblem(F, u, bc, J)


def test_nonlinear_variational_solver_parameters_and_return():
    V, u, problem = nonlinear_problem()
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    assert prm["newton_solver"]["maximum_iterations"] == 50
    prm["newton_solver"]["relative_tolerance"] = 1e-6
    prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-10
    prm["fancy"]["nested"]["key"] = 3
    iterations, converged = solver.solve()
    assert converged and 1 <= iterations <= 10 and solver.iterations == iterations
    # solve again: already converged
    it2, ok2 = solver.solve()
    assert ok2 and it2 <= 1
    # linear solver choice per parameters
    V, u, problem = nonlinear_problem()
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = "gmres"
    solver.parameters["newton_solver"]["preconditioner"] = "jacobi"
    solver.parameters["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1e-12
    it, ok = solver.solve()
    assert ok and it >= 1


def test_nonlinear_relaxation_and_nonconvergence():
    V, u, problem = nonlinear_problem()
    s1 = NonlinearVariationalSolver(problem)
    it1, _ = s1.solve()
    V, u, problem = nonlinear_problem()
    s2 = NonlinearVariationalSolver(problem)
    s2.parameters["newton_solver"]["relaxation_parameter"] = 0.5
    it2, ok = s2.solve()
    assert ok and it2 > it1
    V, u, problem = nonlinear_problem()
    s3 = NonlinearVariationalSolver(problem)
    s3.parameters["newton_solver"]["maximum_iterations"] = 1
    with pytest.raises(ConvergenceError):
        s3.solve()
    V, u, problem = nonlinear_problem()
    s4 = NonlinearVariationalSolver(problem)
    s4.parameters["newton_solver"]["maximum_iterations"] = 1
    s4.parameters["newton_solver"]["error_on_nonconvergence"] = False
    assert s4.solve() == (1, False)


def test_linear_variational_problem_and_solver():
    mesh, V, u_D, bc, a, L = poisson_setup(5)
    ref = Function(V)
    solve(a == L, ref, bc)
    u = Function(V)
    solver = LinearVariationalSolver(LinearVariationalProblem(a, L, u, bc))
    solver.parameters["linear_solver"] = "cg"
    solver.parameters["preconditioner"] = "ilu"
    solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-13
    solver.solve()
    assert np.allclose(vals(u), vals(ref), atol=1e-9)
    u2 = Function(V)
    LinearVariationalSolver(LinearVariationalProblem(a, L, u2, [bc])).solve()
    assert np.allclose(vals(u2), vals(ref))


# --- solver name mapping

@pytest.mark.parametrize("name", ["lu", "default", "mumps", "umfpack", "superlu", "petsc", "LU"])
def test_linear_option_direct(name):
    assert _linear_options(name) == {"ksp_type": "preonly", "pc_type": "lu"}


@pytest.mark.parametrize("solver,ksp", [("cg", "cg"), ("gmres", "gmres"), ("minres", "minres"),
                                        ("bicgstab", "gmres"), ("tfqmr", "gmres")])
@pytest.mark.parametrize("pc,expected", [("none", "none"), ("jacobi", "jacobi"), ("ilu", "jacobi"),
                                         ("icc", "jacobi"), ("sor", "jacobi"), ("amg", "jacobi"),
                                         ("hypre_amg", "jacobi"), ("default", "jacobi"),
                                         ("lu", "lu")])
def test_linear_option_krylov(solver, ksp, pc, expected):
    assert _linear_options(solver, pc) == {"ksp_type": ksp, "pc_type": expected}


def test_linear_option_unknown_solver():
    with pytest.raises(NotImplementedError, match="linear_solver"):
        _linear_options("sparkle")


def test_convert_solver_parameters():
    assert _convert(None) == {"ksp_type": "preonly", "pc_type": "lu"}
    assert _convert({"linear_solver": "cg", "preconditioner": "ilu"}) == \
        {"ksp_type": "cg", "pc_type": "jacobi"}
    out = _convert({"newton_solver": {"relative_tolerance": 1e-6, "absolute_tolerance": 1e-7,
                                      "maximum_iterations": 12, "relaxation_parameter": 0.8,
                                      "linear_solver": "gmres", "preconditioner": "none",
                                      "krylov_solver": {"relative_tolerance": 1e-11,
                                                        "maximum_iterations": 77}}})
    assert out == {"ksp_type": "gmres", "pc_type": "none", "snes_rtol": 1e-6, "snes_atol": 1e-7,
                   "snes_max_it": 12, "snes_damping": 0.8, "ksp_rtol": 1e-11, "ksp_max_it": 77}
    out = _convert({"krylov_solver": {"relative_tolerance": 1e-5}, "linear_solver": "minres"})
    assert out["ksp_rtol"] == 1e-5 and out["ksp_type"] == "minres"
    assert _convert({"ksp_type": "cg", "pc_type": "none"}) == {"ksp_type": "cg", "pc_type": "none"}


@pytest.mark.parametrize("solver,pc", [("lu", "default"), ("mumps", "default"), ("cg", "ilu"),
                                       ("cg", "none"), ("cg", "jacobi"), ("gmres", "default"),
                                       ("gmres", "none"), ("minres", "none"), ("cg", "amg"),
                                       ("bicgstab", "default")])
def test_solver_parameters_give_same_solution(solver, pc):
    mesh, V, u_D, bc, a, L = poisson_setup(6)
    ref = Function(V)
    solve(a == L, ref, bc)
    u = Function(V)
    solve(a == L, u, bc, solver_parameters={"linear_solver": solver, "preconditioner": pc,
                                            "krylov_solver": {"relative_tolerance": 1e-13,
                                                              "maximum_iterations": 500}})
    assert np.allclose(vals(u), vals(ref), atol=1e-8)


def test_solve_rejects_adaptive_arguments():
    mesh, V, u_D, bc, a, L = poisson_setup(2)
    with pytest.raises(NotImplementedError):
        solve(a == L, Function(V), bc, tol=1e-3, M=Constant(1.0) * dx(domain=mesh))


# --- assembled systems

def test_assemble_bc_apply_solve_equals_variational():
    mesh, V, u_D, bc, a, L = poisson_setup(6)
    ref = Function(V)
    solve(a == L, ref, bc)
    A, b = assemble(a), assemble(L)
    bc.apply(A, b)
    u = Function(V)
    solve(A, u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)
    # separate applies
    A, b = assemble(a), assemble(L)
    bc.apply(A)
    bc.apply(b)
    u = Function(V)
    solve(A, u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)
    # solve into a Function directly
    u = Function(V)
    solve(A, u, b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)


@pytest.mark.parametrize("args", [("cg", "ilu"), ("cg", "none"), ("gmres",), ("lu",), ("cg",)])
def test_assembled_solve_with_named_solvers(args):
    mesh, V, u_D, bc, a, L = poisson_setup(5)
    ref = Function(V)
    solve(a == L, ref, bc)
    A, b = assemble(a), assemble(L)
    bc.apply(A, b)
    u = Function(V)
    solve(A, u.vector(), b, *args)
    assert np.allclose(vals(u), vals(ref), atol=1e-7)


def test_assemble_system():
    mesh, V, u_D, bc, a, L = poisson_setup(5)
    ref = Function(V)
    solve(a == L, ref, bc)
    A, b = assemble_system(a, L, bc)
    u = Function(V)
    solve(A, u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)
    A, b = assemble_system(a, L, [bc])
    u = Function(V)
    solve(A, u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)
    A, b = assemble_system(a, L)
    assert not A.bcs


def test_assemble_matrix_matches_native():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "P", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(dot(grad(u), grad(v)) * dx)
    fes = ngs.H1(mesh.ngs, order=1)
    uu, vv = fes.TnT()
    B = ngs.BilinearForm(ngs.grad(uu) * ngs.grad(vv) * ngs.dx).Assemble()
    assert np.allclose(A.array(), B.mat.ToDense().NumPy())
    assert isc(assemble(Constant(2.0) * dx(domain=mesh)), 2.0)
    assert isinstance(assemble(u * v * dx), Matrix) and isinstance(assemble(v * dx), Vector)


def test_assemble_tensor_argument():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "P", 1)
    u, v = TrialFunction(V), TestFunction(V)
    b = assemble(Constant(1.0) * v * dx)
    b2 = assemble(Constant(2.0) * v * dx, tensor=b)
    assert b2 is b and np.allclose(b.get_local(), assemble(Constant(2.0) * v * dx).get_local())
    A = assemble(u * v * dx)
    A2 = assemble(2 * u * v * dx, tensor=A)
    assert A2 is A and np.allclose(A.array(), assemble(2 * u * v * dx).array())
    with pytest.raises(TypeError):
        assemble(Constant(1.0) * dx(domain=mesh), tensor=b)


def test_lu_and_krylov_solver_objects():
    mesh, V, u_D, bc, a, L = poisson_setup(5)
    ref = Function(V)
    solve(a == L, ref, bc)
    A, b = assemble_system(a, L, bc)
    u = Function(V)
    LUSolver(A).solve(u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)
    u = Function(V)
    s = LUSolver()
    s.set_operator(A)
    s.solve(u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)
    u = Function(V)
    LUSolver("mumps").solve(A, u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-12)
    u = Function(V)
    ks = KrylovSolver("cg", "ilu")
    ks.parameters["relative_tolerance"] = 1e-13
    ks.solve(A, u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-9)
    u = Function(V)
    ks = KrylovSolver(A, "gmres", "none")
    ks.solve(u.vector(), b)
    assert np.allclose(vals(u), vals(ref), atol=1e-7)


def test_norms():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "P", 1)
    u = interpolate(Expression("x[0]", degree=1), V)
    assert isc(norm(u), math.sqrt(1 / 3)) and isc(norm(u, "L2"), math.sqrt(1 / 3))
    assert isc(norm(u, "H10"), 1.0) and isc(norm(u, "H1"), math.sqrt(1 / 3 + 1))
    vec = u.vector()
    assert isc(norm(vec, "linf"), 1.0) and isc(norm(vec, "l2"), np.linalg.norm(vec.get_local()))
    assert isc(norm(vec, "l1"), vec.get_local().sum()) and isc(norm(u, "l2"), norm(vec, "l2"))
    assert isc(norm(u, "L2", mesh), norm(u))
    with pytest.raises(ValueError):
        norm(u, "L7x")


# ---------------------------------------------------------------- the classic applications

def test_linear_elasticity_clamped_beam_2d_against_native():
    L_, W_ = 4.0, 1.0
    mesh = RectangleMesh(Point(0, 0), Point(L_, W_), 16, 4)
    V = VectorFunctionSpace(mesh, "P", 1)
    tol = 1e-14
    clamped_boundary = lambda x, on_boundary: on_boundary and x[0] < tol
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    rho, g = 0.02, 0.1
    mu, lambda_ = 1.0, 1.25
    f = Constant((0, -rho * g))
    T = Constant((0, 0))

    def epsilon(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

    def sigma(u):
        return lambda_ * nabla_div(u) * Identity(d) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    d = u.geometric_dimension()
    v = TestFunction(V)
    a = inner(sigma(u), epsilon(v)) * dx
    L = dot(f, v) * dx + dot(T, v) * ds
    u = Function(V)
    solve(a == L, u, bc)
    assert d == 2
    uv = u.compute_vertex_values(mesh).reshape(2, -1)
    assert uv[1].min() < -1e-3
    # native NGSolve
    fes = ngs.VectorH1(mesh.ngs, order=1, dirichlet="left")
    uu, vv = fes.TnT()
    eps = lambda w: ngs.Sym(ngs.Grad(w))
    sig = lambda w: lambda_ * ngs.Trace(eps(w)) * ngs.Id(2) + 2 * mu * eps(w)
    A = ngs.BilinearForm(ngs.InnerProduct(sig(uu), eps(vv)) * ngs.dx).Assemble()
    fv = ngs.LinearForm(ngs.CF((0, -rho * g)) * vv * ngs.dx).Assemble()
    gf = ngs.GridFunction(fes)
    gf.vec.data = A.mat.Inverse(fes.FreeDofs()) * fv.vec
    X = mesh.coordinates()
    nat = np.array([gf(mesh.ngs(*p)) for p in X]).T
    assert np.allclose(uv, nat, atol=1e-12)
    # stress as a symmetric matrix, von Mises from the deviator
    s = sigma(u)
    s_sym = assemble(inner(s - s.T, s - s.T) * dx)
    assert s_sym < 1e-24
    von_mises = sqrt(3.0 / 2 * inner(dev(s), dev(s)))
    Vs = FunctionSpace(mesh, "DG", 0)
    assert project(von_mises, Vs).vector().max() > 0


def test_linear_elasticity_beam_3d_symmetry_and_deflection():
    mesh = BoxMesh(Point(0, 0, 0), Point(2, 0.5, 0.5), 8, 2, 2)
    V = VectorFunctionSpace(mesh, "P", 1)
    bc = DirichletBC(V, Constant((0, 0, 0)), "on_boundary && near(x[0], 0)")
    mu, lambda_, rho, g = 1.0, 1.25, 0.02, 0.5
    f = Constant((0, 0, -rho * g))
    u, v = TrialFunction(V), TestFunction(V)
    d = u.geometric_dimension()
    assert d == 3

    def epsilon(w):
        return sym(nabla_grad(w))

    def sigma(w):
        return lambda_ * nabla_div(w) * Identity(d) + 2 * mu * epsilon(w)

    uh = Function(V)
    solve(inner(sigma(u), epsilon(v)) * dx == dot(f, v) * dx, uh, bc)
    uv = uh.compute_vertex_values(mesh).reshape(3, -1)
    X = mesh.coordinates()
    tip = X[:, 0] > 2 - 1e-12
    assert uv[2][tip].mean() < -1e-4
    assert abs(uv[1][tip]).max() < abs(uv[2][tip]).max()
    assert np.allclose(uv[:, X[:, 0] < 1e-12], 0)


def test_taylor_hood_stokes_poiseuille_exact():
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), 6, 6)
    P2 = VectorElement("P", mesh.ufl_cell(), 2)
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, P2 * P1)
    inflow = Expression(("4*x[1]*(1 - x[1])", "0"), degree=2)
    walls = lambda x, on_boundary: on_boundary and (near(x[1], 0) or near(x[1], 1))
    bcs = [DirichletBC(W.sub(0), inflow, "on_boundary && near(x[0], 0)"),
           DirichletBC(W.sub(0), Constant((0, 0)), walls)]
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    a = inner(grad(u), grad(v)) * dx - div(v) * p * dx + q * div(u) * dx
    L = dot(Constant((0, 0)), v) * dx
    w = Function(W)
    solve(a == L, w, bcs)
    uh, ph = w.split(deepcopy=True)
    exact_u = Expression(("4*x[1]*(1 - x[1])", "0"), degree=2)
    exact_p = Expression("8*(1 - x[0])", degree=1)
    assert errornorm(exact_u, uh) < 1e-10
    assert errornorm(exact_p, ph) < 1e-10
    assert abs(assemble(div(uh) ** 2 * dx)) < 1e-18
    # the same with the mixed unknown accessed through split
    U, P = split(w)
    assert isc(assemble(div(U) ** 2 * dx), 0.0, atol=1e-18)
    assert isc(assemble(P * dx), 4.0, rtol=1e-9)


def test_stokes_direct_krylov_agree():
    mesh = UnitSquareMesh(4, 4)
    W = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), 2) * FiniteElement("P", mesh.ufl_cell(), 1))
    bcs = [DirichletBC(W.sub(0), Constant((1.0, 0.0)), "on_boundary && near(x[1], 1)"),
           DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary && x[1] < 1 - DOLFIN_EPS")]
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    # penalised pressure makes the system positive definite for gmres without pinning
    a = inner(grad(u), grad(v)) * dx - div(v) * p * dx + q * div(u) * dx + 1e-3 * p * q * dx
    L = dot(Constant((0, 0)), v) * dx
    w1, w2 = Function(W), Function(W)
    solve(a == L, w1, bcs)
    solve(a == L, w2, bcs, solver_parameters={"linear_solver": "gmres", "preconditioner": "none",
                                              "krylov_solver": {"relative_tolerance": 1e-12,
                                                                "maximum_iterations": 2000}})
    assert np.allclose(vals(w1), vals(w2), atol=1e-6)


# ---------------------------------------------------------------- output

def test_plot_scalar_function_returns_artist_for_colorbar():
    import matplotlib.pyplot as plt
    mesh, V, u_D, bc, a, L = poisson_setup(4)
    u = Function(V)
    solve(a == L, u, bc)
    plt.figure()
    c = plot(u, title="solution")
    assert plt.gca().get_title() == "solution" and len(plt.gcf().axes) == 1
    cb = plt.colorbar(c)
    assert cb is not None and len(plt.gcf().axes) == 2
    plt.close("all")


def test_plot_draws_on_current_axes_and_without_colorbar():
    import matplotlib.pyplot as plt
    mesh, V, u_D, bc, a, L = poisson_setup(3)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax2)
    plot(interpolate(u_D, V))
    assert len(fig.axes) == 2 and len(ax2.collections) >= 1 and not ax1.collections
    plot(mesh)
    assert len(ax2.lines) + len(ax2.collections) > 1
    plt.close("all")


def test_plot_mesh_expression_vector_and_modes():
    import matplotlib.pyplot as plt
    mesh = UnitSquareMesh(4, 4)
    plt.figure()
    art = plot(mesh, title="mesh")
    assert art is not None and plt.gca().get_title() == "mesh"
    V = VectorFunctionSpace(mesh, "P", 1)
    f = interpolate(Expression(("x[1]", "-x[0]"), degree=1), V)
    plt.figure()
    q_ = plot(f, title="field")
    assert hasattr(q_, "U")  # a quiver
    plt.figure()
    plot(f, mode="color")
    plt.figure()
    plot(0.1 * f, mode="displacement")
    plt.figure()
    plot(Expression("x[0]*x[1]", degree=2))
    plt.figure()
    plot(Function(FunctionSpace(mesh, "P", 1)), levels=5)
    plt.close("all")


def test_plot_1d_and_ignored_keywords():
    import matplotlib.pyplot as plt
    mesh = UnitIntervalMesh(6)
    V = FunctionSpace(mesh, "P", 1)
    u = interpolate(Expression("x[0]*x[0]", degree=2), V)
    plt.figure()
    line = plot(u, title="1d", interactive=False, scalarbar=True, backend="matplotlib", color="r")
    assert line.get_xdata().min() == 0.0 and isc(line.get_ydata().max(), 1.0)
    plt.close("all")


def test_plot_3d_surface_smoke():
    import matplotlib.pyplot as plt
    mesh = UnitCubeMesh(2, 2, 2)
    V = FunctionSpace(mesh, "P", 1)
    plt.figure()
    ax = plot(interpolate(Expression("x[0]", degree=1), V))
    assert ax is not None
    with pytest.raises(NotImplementedError):
        plot(mesh)
    plt.close("all")


def test_file_pvd_output(tmp_path):
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "P", 1)
    u = interpolate(Expression("x[0]", degree=1), V)
    u.rename("u", "")
    out = tmp_path / "sub" / "solution.pvd"
    vtk = File(str(out))
    vtk << u
    files = sorted(p.name for p in (tmp_path / "sub").iterdir())
    assert any(f.endswith(".vtu") for f in files)
    series = File(str(tmp_path / "series.pvd"))
    for t in (0.0, 0.5, 1.0):
        series << (u, t)
    assert len(list(tmp_path.glob("series*.vtu"))) >= 3
    f2 = File(str(tmp_path / "vec.pvd"))
    f2 << interpolate(Expression(("x[0]", "x[1]"), degree=1), VectorFunctionSpace(mesh, "P", 1))
    assert list(tmp_path.glob("vec*.vtu"))
    with pytest.raises(NotImplementedError):
        File(str(tmp_path / "m.xml"))
    with pytest.raises(NotImplementedError):
        File(str(tmp_path / "mesh.pvd")) << mesh
    with pytest.raises(TypeError):
        File(str(tmp_path / "x.pvd")) << 3.0


# ---------------------------------------------------------------- misc and helpful errors

def test_parameters_are_permissive_nested_dict():
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["optimize"] = True
    parameters["allow_extrapolation"] = True
    parameters["new"]["deeper"]["still"] = 1
    assert parameters["new"]["deeper"]["still"] == 1 and parameters.allow_extrapolation is True
    parameters["linear_algebra_backend"] = "PETSc"
    assert parameters["reorder_dofs_serial"] in (True, False)


def test_log_functions(capsys):
    set_log_level(LogLevel.ERROR)
    info("hidden")
    set_log_level(LogLevel.INFO)
    info("shown")
    set_log_level(30)
    set_log_active(False)
    info("hidden too")
    set_log_active(True)
    out = capsys.readouterr().out
    assert "shown" in out and "hidden" not in out
    assert LogLevel.ERROR > LogLevel.WARNING > LogLevel.INFO > LogLevel.PROGRESS > LogLevel.DEBUG
    assert all(hasattr(LogLevel, n) for n in ("ERROR", "WARNING", "INFO", "PROGRESS", "DEBUG"))
    interactive()


def test_mpi_free():
    assert MPI.rank(mpi_comm_world()) == 0 and MPI.size(MPI.comm_world) == 1


@pytest.mark.parametrize("name,args", [
    ("MeshFunction", ("size_t", None, 1)), ("XDMFFile", ("a.xdmf",)), ("refine", (None,)),
    ("TimeSeries", ("x",)), ("MeshValueCollection", ()), ("HDF5File", ()),
    ("vertex_to_dof_map", ()), ("FacetFunction", ()), ("SubMesh", ()), ("FunctionAssigner", ())])
def test_unsupported_names_give_helpful_error(name, args):
    import ngsufl.fenics as fe
    obj = getattr(fe, name)
    with pytest.raises(NotImplementedError, match="not available in ngsufl.fenics yet"):
        obj(*args)


def test_unsupported_names_exported_by_star_import():
    ns = {}
    exec("from ngsufl.fenics import *", ns)
    for name in ("MeshFunction", "XDMFFile", "UserExpression", "refine", "TimeSeries"):
        assert name in ns


def test_user_expression_subclass_fails_clearly():
    with pytest.raises(NotImplementedError, match="UserExpression"):
        class MyExpr(UserExpression):
            def eval(self, values, x):
                values[0] = x[0]


def test_module_getattr():
    import ngsufl.fenics as fe
    with pytest.raises(NotImplementedError, match="mshr"):
        fe.mshr.Rectangle(None, None)
    with pytest.raises(AttributeError):
        fe.surely_not_there


def test_all_core_names_reexported():
    import ngsufl
    import ngsufl.fenics as fe
    assert set(ngsufl.__all__) <= set(fe.__all__)
    assert fe.DirichletBC is not ngsufl.DirichletBC and fe.UnitSquareMesh is not ngsufl.UnitSquareMesh
    assert fe.dx is ngsufl.dx and fe.Constant is ngsufl.Constant


def test_constants_and_ufl_names_work():
    mesh = UnitSquareMesh(2, 2)
    c = Constant(3.0)
    assert isc(float(c), 3.0) and isc(assemble(c * dx(domain=mesh)), 3.0)
    c.assign(5.0)
    assert isc(assemble(c * dx(domain=mesh)), 5.0)
    vc = Constant((1.0, 2.0))
    assert np.allclose(vc.value, (1, 2))
    n = FacetNormal(mesh)
    assert isc(assemble(dot(Constant((1.0, 0.0)), n) * ds(domain=mesh)), 0.0, atol=1e-14)
    h = CellDiameter(mesh)
    assert isc(assemble(h * dx(domain=mesh)), math.sqrt(2) / 2)


def test_ds_integral_over_boundary_and_dx_degree():
    mesh = UnitSquareMesh(4, 4)
    assert isc(assemble(Constant(1.0) * ds(domain=mesh)), 4.0)
    x = SpatialCoordinate(mesh)
    assert isc(assemble(x[0] ** 3 * dx(degree=3)), 0.25)


# ---------------------------------------------------------------- documented limitations

def test_gradient_of_expression_not_supported():
    mesh = UnitSquareMesh(3, 3)
    e = Expression(("x[0]*x[0]", "x[1]"), degree=2, mesh=mesh)
    assert isc(assemble(div(e) * dx), 1.0 + 1.0)

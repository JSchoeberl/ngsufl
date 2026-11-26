import math
import os

import numpy as np
import ngsolve as ngs
import pytest

from ngsufl import *
from conftest import approx

isc = np.isclose


def poisson_dirichlet(V, g, bcs, f=None):
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    f = Constant(V.mesh, 0.0) if f is None else f
    solve(inner(grad(u), grad(v)) * dx == f * v * dx, uh, bcs)
    return uh


@pytest.fixture(scope="module")
def sq():
    return UnitSquareMesh(4, 4)


# ---------- Dirichlet BC selectors

LR = {
    "name": lambda m: "left|right",
    "regexp": lambda m: "l.*|r.*",
    "id_tuple": lambda m: (1, 2),
    "ids_names": lambda m: m.boundary_ids("left|right"),
    "vectorised": lambda m: (lambda x: isc(x[0], 0.0) | isc(x[0], 1.0)),
    "legacy": lambda m: (lambda x, on_boundary: on_boundary and (abs(x[0]) < 1e-12 or abs(x[0] - 1) < 1e-12)),
}


@pytest.mark.parametrize("kind", list(LR))
@pytest.mark.parametrize("quad", [False, True])
def test_bc_where_kinds_partial(sq, kind, quad):
    m = UnitSquareMesh(4, 4, quadrilateral=quad)
    V = FunctionSpace(m, "Lagrange", 2)
    x = SpatialCoordinate(m)
    bc = DirichletBC(V, 1 + 2 * x[0], LR[kind](m))
    assert sum(bc.dofs()) == 2 * (2 * 4 + 1)
    uh = poisson_dirichlet(V, None, bc)
    assert errornorm(1 + 2 * x[0], uh) < 1e-12


def test_bc_where_on_boundary_and_single_id_name(sq):
    V = FunctionSpace(sq, "Lagrange", 2)
    x = SpatialCoordinate(sq)
    g = 1 + x[0] ** 2 - 0.5 * x[1] ** 2
    f = -div(grad(g))
    uh = poisson_dirichlet(V, g, DirichletBC(V, g, "on_boundary"), f)
    assert errornorm(g, uh) < 1e-12
    uh = poisson_dirichlet(V, g, DirichletBC(V, g), f)
    assert errornorm(g, uh) < 1e-12
    bcs = [DirichletBC(V, g, i) for i in (1, 2, 3, 4)]
    assert errornorm(g, poisson_dirichlet(V, g, bcs, f)) < 1e-12
    bcs = [DirichletBC(V, g, n) for n in ("bottom", "right", "top", "left")]
    assert errornorm(g, poisson_dirichlet(V, g, bcs, f)) < 1e-12
    bcs = [DirichletBC(V, g, (1, 2)), DirichletBC(V, g, (3, 4))]
    assert errornorm(g, poisson_dirichlet(V, g, bcs, f)) < 1e-12


def test_bc_dof_counts_and_values(sq):
    V = FunctionSpace(sq, "Lagrange", 1)
    for where, n in [("left", 5), ("left|right", 10), ("on_boundary", 16), (1, 5), ((1, 2), 10)]:
        assert sum(DirichletBC(V, 0.0, where).dofs()) == n
    V2 = FunctionSpace(sq, "Lagrange", 2)
    assert sum(DirichletBC(V2, 0.0, "left").dofs()) == 5 + 4
    assert sum(DirichletBC(V2, 0.0, "on_boundary").dofs()) == 16 + 16


def test_bc_apply_values(sq):
    V = FunctionSpace(sq, "Lagrange", 1)
    gf = Function(V)
    DirichletBC(V, 3.0, "bottom").apply(gf.gf)
    assert gf.at(0.5, 0.0) == approx(3.0)
    assert gf.at(0.5, 0.5) == approx(0.0)
    DirichletBC(V, lambda x: x[0], "top").apply(gf.gf)
    assert gf.at(0.25, 1.0) == approx(0.25)


def test_bc_unknown_name_and_bad_id(sq):
    V = FunctionSpace(sq, "Lagrange", 1)
    with pytest.raises(Exception):
        DirichletBC(V, 0.0, 17).dofs()
    bc = DirichletBC(V, 0.0, "nonexistent")
    assert sum(bc.dofs()) == 0


G_KINDS = {
    "number": lambda m, V: (2.5, lambda p: 2.5),
    "ufl": lambda m, V: (1 + SpatialCoordinate(m)[0] + 2 * SpatialCoordinate(m)[1], lambda p: 1 + p[0] + 2 * p[1]),
    "function": lambda m, V: (Function(V).interpolate(lambda x: 1 + x[0] + 2 * x[1]), lambda p: 1 + p[0] + 2 * p[1]),
    "constant": lambda m, V: (Constant(m, 4.0), lambda p: 4.0),
    "callable": lambda m, V: (lambda x: 1 + x[0] + 2 * x[1], lambda p: 1 + p[0] + 2 * p[1]),
    "ngs_cf": lambda m, V: (1 + ngs.x + 2 * ngs.y, lambda p: 1 + p[0] + 2 * p[1]),
    "callable_np": lambda m, V: (lambda x: 1 + np.sin(x[0]) * 0 + x[0] + 2 * x[1], lambda p: 1 + p[0] + 2 * p[1]),
}


@pytest.mark.parametrize("kind", list(G_KINDS))
def test_bc_g_kinds(sq, kind):
    V = FunctionSpace(sq, "Lagrange", 2)
    g, ref = G_KINDS[kind](sq, V)
    gf = Function(V)
    DirichletBC(V, g, "on_boundary").apply(gf.gf)
    for p in [(0.0, 0.3), (1.0, 0.7), (0.4, 0.0), (0.6, 1.0)]:
        assert gf.at(*p) == approx(ref(p), abs=1e-12)
    assert gf.at(0.5, 0.5) == approx(0.0)


def test_bc_constant_changes_between_solves(sq):
    V = FunctionSpace(sq, "Lagrange", 1)
    c = Constant(sq, 1.0)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    bc = DirichletBC(V, c, "on_boundary")
    solve(inner(grad(u), grad(v)) * dx == Constant(sq, 0.0) * v * dx, uh, bc)
    assert uh.at(0.5, 0.5) == approx(1.0)
    c.assign(3.0)
    solve(inner(grad(u), grad(v)) * dx == Constant(sq, 0.0) * v * dx, uh, bc)
    assert uh.at(0.5, 0.5) == approx(3.0)


def test_bc_vector_components_subspaces():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    uex = as_vector((1 + SpatialCoordinate(m)[0], 2 - SpatialCoordinate(m)[1]))
    bcs = [DirichletBC(V.sub(0), 1 + SpatialCoordinate(m)[0], "left|right"),
           DirichletBC(V.sub(1), 2 - SpatialCoordinate(m)[1], "top|bottom")]
    n0 = FunctionSpace(m, "Lagrange", 2).dim()
    assert V.sub(0).dof_range() == range(0, n0)
    assert sum(bcs[0].dofs()) == 9 * 2
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == dot(Constant(m, (0.0, 0.0)), v) * dx, uh, bcs)
    assert errornorm(uex, uh) < 1e-12


def test_bc_vector_whole_space_tuple_and_callable():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(m)
    uex = as_vector((1 + x[0], 2 - x[1]))
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == dot(Constant(m, (0.0, 0.0)), v) * dx, uh, DirichletBC(V, uex, "on_boundary"))
    assert errornorm(uex, uh) < 1e-12
    uh2 = Function(V)
    solve(inner(grad(u), grad(v)) * dx == dot(Constant(m, (0.0, 0.0)), v) * dx, uh2,
          DirichletBC(V, lambda x: (1 + x[0], 2 - x[1]), "on_boundary"))
    assert errornorm(uex, uh2) < 1e-12
    gf = Function(V)
    DirichletBC(V, (1.0, 2.0), "left").apply(gf.gf)
    assert gf.at(0.0, 0.5) == approx([1.0, 2.0])
    DirichletBC(V, 7.0, "right").apply(gf.gf)
    assert gf.at(1.0, 0.5) == approx([7.0, 7.0])


def test_bc_on_mixed_subspaces_stokes():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    R = FunctionSpace(m, "Real", 0)
    W = V * Q * R
    (u, p, c), (v, q, r) = TrialFunctions(W), TestFunctions(W)
    x = SpatialCoordinate(m)
    uex = as_vector((x[1] ** 2, 0 * x[0]))
    pex = x[0] - 0.5
    a = inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx + c * q * dx + r * p * dx
    f = -div(grad(uex)) + grad(pex)
    L = inner(f, v) * dx
    w = Function(W)
    solve(a == L, w, DirichletBC(W.sub(0), uex, "on_boundary"))
    uh, ph, ch = w.subfunctions
    assert errornorm(uex, uh) < 1e-11
    assert errornorm(pex, ph) < 1e-11
    assert norm(div(uh)) < 1e-11
    assert abs(ch.array[0]) < 1e-10


def test_bc_on_subspace_component_of_mixed():
    m = UnitSquareMesh(3, 3)
    V = VectorFunctionSpace(m, "Lagrange", 1)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    bc = DirichletBC(W.sub(0).sub(1), 5.0, "left")
    nV, nQ = V.dim(), Q.dim()
    dofs = [i for i, b in enumerate(bc.dofs()) if b]
    n0 = nV // 2
    assert len(dofs) == 4
    assert all(n0 <= d < nV for d in dofs)
    w = Function(W)
    bc.apply(w.gf)
    assert w.sub(0).at(0.0, 0.5) == approx([0.0, 5.0])


def _stokes_pin(where):
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    x = SpatialCoordinate(m)
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
    uex = as_vector((x[1] ** 2, 0 * x[0]))
    pex = x[0]
    a = inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx
    L = inner(-div(grad(uex)) + grad(pex), v) * dx
    bc_p = DirichletBC(W.sub(1), pex, where)
    w = Function(W)
    solve(a == L, w, [DirichletBC(W.sub(0), uex, "on_boundary"), bc_p])
    return w, uex, pex, bc_p


def test_bc_pressure_on_boundary_part():
    w, uex, pex, bc = _stokes_pin("bottom")
    assert errornorm(uex, w.subfunctions[0]) < 1e-11
    assert errornorm(pex, w.subfunctions[1]) < 1e-11


def test_bc_pressure_on_boundary_part_callable():
    w, uex, pex, bc = _stokes_pin(lambda x: isc(x[1], 0.0))
    assert sum(bc.dofs()) == 5
    assert errornorm(pex, w.subfunctions[1]) < 1e-11


@pytest.mark.parametrize("legacy", [False, True])
def test_bc_pointwise_pin_callable(legacy):
    if legacy:
        where = lambda x, on_boundary: abs(x[0]) < 1e-12 and abs(x[1]) < 1e-12
    else:
        where = lambda x: isc(x[0], 0.0) & isc(x[1], 0.0)
    m = UnitSquareMesh(4, 4)
    Q = FunctionSpace(m, "Lagrange", 1)
    assert sum(DirichletBC(Q, 0.0, where).dofs()) == 1


# ---------- linear solver parameters

def _poisson_problem(m):
    V = FunctionSpace(m, "Lagrange", 2)
    x = SpatialCoordinate(m)
    g = 1 + x[0] ** 2 + 2 * x[1] ** 2
    u, v = TrialFunction(V), TestFunction(V)
    return V, g, inner(grad(u), grad(v)) * dx, Constant(m, -6.0) * v * dx


PARAMS = [
    {}, {"ksp_type": "preonly", "pc_type": "lu"}, {"ksp_type": "preonly", "pc_type": "cholesky"},
    {"ksp_type": "preonly", "pc_type": "sparsecholesky"},
    {"ksp_type": "cg", "pc_type": "jacobi"}, {"ksp_type": "cg", "pc_type": "none"},
    {"ksp_type": "cg", "pc_type": "lu"}, {"ksp_type": "cg", "pc_type": "cholesky"},
    {"ksp_type": "gmres", "pc_type": "jacobi"}, {"ksp_type": "gmres", "pc_type": "none"},
    {"ksp_type": "gmres", "pc_type": "lu"}, {"ksp_type": "minres", "pc_type": "jacobi"},
    {"ksp_type": "minres", "pc_type": "none"},
    {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12, "ksp_max_it": 500, "ksp_monitor": True},
]


@pytest.mark.parametrize("par", PARAMS, ids=lambda p: "-".join(f"{k}={v}" for k, v in p.items()) or "default")
def test_solver_parameters(sq, par):
    V, g, a, L = _poisson_problem(sq)
    uh = Function(V)
    solve(a == L, uh, DirichletBC(V, g, "on_boundary"), solver_parameters=par)
    assert errornorm(g, uh) < 1e-8


@pytest.mark.parametrize("pc", ["umfpack", "pardiso", "mumps"])
def test_optional_direct_solvers(sq, pc):
    V, g, a, L = _poisson_problem(sq)
    uh = Function(V)
    try:
        solve(a == L, uh, DirichletBC(V, g, "on_boundary"), solver_parameters={"pc_type": pc})
    except Exception as e:
        pytest.skip(f"{pc} not available: {type(e).__name__}")
    assert errornorm(g, uh) < 1e-10


def test_solver_parameters_unknown_options(sq):
    V, g, a, L = _poisson_problem(sq)
    uh = Function(V)
    bc = DirichletBC(V, g, "on_boundary")
    with pytest.raises(NotImplementedError):
        solve(a == L, uh, bc, solver_parameters={"ksp_type": "bcgs"})
    with pytest.raises(NotImplementedError):
        solve(a == L, uh, bc, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"})
    with pytest.raises(NotImplementedError):
        solve(a == L, uh, bc, solver_parameters={"pc_type": "ilu"})


def test_ksp_max_it_limits_iterations(sq, capsys):
    V, g, a, L = _poisson_problem(sq)
    uh = Function(V)
    solve(a == L, uh, DirichletBC(V, g, "on_boundary"),
          solver_parameters={"ksp_type": "cg", "pc_type": "jacobi", "ksp_max_it": 2, "ksp_rtol": 1e-14})
    assert errornorm(g, uh) > 1e-8


def test_ksp_rtol_controls_accuracy(sq):
    V, g, a, L = _poisson_problem(sq)
    errs = []
    for tol in (1e-1, 1e-6, 1e-12):
        uh = Function(V)
        solve(a == L, uh, DirichletBC(V, g, "on_boundary"),
              solver_parameters={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": tol})
        errs.append(errornorm(g, uh))
    assert errs[0] > errs[1] > errs[2] or errs[2] < 1e-9


def test_gmres_nonsymmetric_convection_diffusion(sq):
    V = FunctionSpace(sq, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    b = Constant(sq, (1.0, 0.5))
    x = SpatialCoordinate(sq)
    g = 1 + x[0] + x[1]
    a = 0.1 * inner(grad(u), grad(v)) * dx + dot(b, grad(u)) * v * dx
    L = dot(b, grad(g)) * v * dx
    ref, uh = Function(V), Function(V)
    bc = DirichletBC(V, g, "on_boundary")
    solve(a == L, ref, bc)
    solve(a == L, uh, bc, solver_parameters={"ksp_type": "gmres", "pc_type": "jacobi", "ksp_rtol": 1e-12})
    assert errornorm(g, ref) < 1e-10
    assert np.abs(uh.array - ref.array).max() < 1e-8


def test_solve_without_bcs_and_zero_rhs_returns_none(sq):
    V = FunctionSpace(sq, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    r = solve(u * v * dx + inner(grad(u), grad(v)) * dx == Constant(sq, 1.0) * v * dx, uh)
    assert r is None
    assert uh.at(0.5, 0.5) == approx(1.0, abs=2e-2)
    uh2 = Function(V)
    solve(u * v * dx == Constant(sq, 0.0) * v * dx, uh2)
    assert np.abs(uh2.array).max() < 1e-14


def test_solve_nonzero_initial_guess_is_overwritten(sq):
    V, g, a, L = _poisson_problem(sq)
    uh = Function(V).interpolate(lambda x: 100 * x[0])
    solve(a == L, uh, DirichletBC(V, g, "on_boundary"))
    assert errornorm(g, uh) < 1e-10


def test_solve_bc_list_tuple_single(sq):
    V, g, a, L = _poisson_problem(sq)
    bc = DirichletBC(V, g, "on_boundary")
    for bcs in (bc, [bc], (bc,)):
        uh = Function(V)
        solve(a == L, uh, bcs)
        assert errornorm(g, uh) < 1e-10


def test_solve_wrong_equation_type(sq):
    V = FunctionSpace(sq, "Lagrange", 1)
    with pytest.raises(TypeError):
        solve(TrialFunction(V) * TestFunction(V) * dx, Function(V))


def test_solve_bcs_keyword_and_flags(sq):
    V, g, a, L = _poisson_problem(sq)
    uh = Function(V)
    solve(a == L, uh, bcs=DirichletBC(V, g, "on_boundary"), solver_parameters={"pc_type": "lu"})
    assert errornorm(g, uh) < 1e-10


# ---------- convergence

def _sin_data(m):
    x = SpatialCoordinate(m)
    uex = sin(pi * x[0]) * sin(pi * x[1])
    return uex, 2 * pi ** 2 * uex


def _rates(errs, hs):
    return [math.log(errs[i] / errs[i + 1]) / math.log(hs[i] / hs[i + 1]) for i in range(len(errs) - 1)]


@pytest.mark.parametrize("k,ns", [(1, (4, 8, 16)), (2, (4, 8, 16)), (3, (2, 4, 8))])
def test_poisson_convergence(k, ns):
    e0, e1, hs = [], [], []
    for n in ns:
        m = UnitSquareMesh(n, n)
        V = FunctionSpace(m, "Lagrange", k)
        uex, f = _sin_data(m)
        u, v = TrialFunction(V), TestFunction(V)
        uh = Function(V)
        solve(inner(grad(u), grad(v)) * dx == f * v * dx, uh, DirichletBC(V, 0.0, "on_boundary"))
        e0.append(errornorm(uex, uh, "L2"))
        e1.append(errornorm(uex, uh, "H1"))
        hs.append(1.0 / n)
    assert min(_rates(e0, hs)) > k + 1 - 0.3
    assert min(_rates(e1, hs)) > k - 0.2
    assert e0[-1] < e1[-1]


def test_poisson_convergence_quads_and_3d():
    errs, hs = [], []
    for n in (4, 8, 16):
        m = UnitSquareMesh(n, n, quadrilateral=True)
        V = FunctionSpace(m, "Q", 2)
        uex, f = _sin_data(m)
        u, v = TrialFunction(V), TestFunction(V)
        uh = Function(V)
        solve(inner(grad(u), grad(v)) * dx == f * v * dx, uh, DirichletBC(V, 0.0, "on_boundary"))
        errs.append(errornorm(uex, uh))
        hs.append(1.0 / n)
    assert min(_rates(errs, hs)) > 2.7
    errs, hs = [], []
    for n in (2, 4):
        m = UnitCubeMesh(n, n, n)
        V = FunctionSpace(m, "Lagrange", 2)
        x = SpatialCoordinate(m)
        uex = sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])
        u, v = TrialFunction(V), TestFunction(V)
        uh = Function(V)
        solve(inner(grad(u), grad(v)) * dx == 3 * pi ** 2 * uex * v * dx, uh, DirichletBC(V, 0.0, "on_boundary"))
        errs.append(errornorm(uex, uh))
        hs.append(1.0 / n)
    assert min(_rates(errs, hs)) > 2.3


def test_poisson_neumann_data_via_ds():
    errs, hs = [], []
    for n in (4, 8, 16):
        m = UnitSquareMesh(n, n)
        V = FunctionSpace(m, "Lagrange", 2)
        x = SpatialCoordinate(m)
        uex = cos(pi * x[0]) * x[1] ** 2 + 1
        f = -div(grad(uex)) + uex
        nn = FacetNormal(m)
        u, v = TrialFunction(V), TestFunction(V)
        a = (inner(grad(u), grad(v)) + u * v) * dx
        L = f * v * dx + dot(grad(uex), nn) * v * ds
        uh = Function(V)
        solve(a == L, uh)
        errs.append(errornorm(uex, uh))
        hs.append(1.0 / n)
    assert min(_rates(errs, hs)) > 2.6


def test_mixed_poisson_convergence():
    eu, es, hs = [], [], []
    for n in (4, 8, 16):
        m = UnitSquareMesh(n, n)
        W = FunctionSpace(m, "RT", 2) * FunctionSpace(m, "DG", 1)
        (sg, u), (tau, v) = TrialFunctions(W), TestFunctions(W)
        x = SpatialCoordinate(m)
        uex = sin(pi * x[0]) * sin(pi * x[1])
        f = 2 * pi ** 2 * uex
        a = (dot(sg, tau) + div(tau) * u + div(sg) * v) * dx
        L = -f * v * dx
        w = Function(W)
        solve(a == L, w)
        sh, uh = w.subfunctions
        eu.append(errornorm(uex, uh))
        es.append(errornorm(grad(uex), sh))
        hs.append(1.0 / n)
    assert min(_rates(eu, hs)) > 1.8
    assert min(_rates(es, hs)) > 1.8


def test_mixed_poisson_bdm_dg0():
    eu, hs = [], []
    for n in (4, 8, 16):
        m = UnitSquareMesh(n, n)
        W = FunctionSpace(m, "BDM", 1) * FunctionSpace(m, "DG", 0)
        (sg, u), (tau, v) = TrialFunctions(W), TestFunctions(W)
        x = SpatialCoordinate(m)
        uex = sin(pi * x[0]) * sin(pi * x[1])
        a = (dot(sg, tau) + div(tau) * u + div(sg) * v) * dx
        L = -2 * pi ** 2 * uex * v * dx
        w = Function(W)
        solve(a == L, w)
        eu.append(errornorm(uex, w.subfunctions[1]))
        hs.append(1.0 / n)
    assert min(_rates(eu, hs)) > 0.9


def test_maxwell_hcurl_convergence():
    errs, hs = [], []
    for n in (2, 4):
        m = UnitCubeMesh(n, n, n)
        V = FunctionSpace(m, "N1curl", 2)
        x = SpatialCoordinate(m)
        uex = as_vector((sin(pi * x[1]), sin(pi * x[2]), sin(pi * x[0])))
        f = curl(curl(uex)) + uex
        u, v = TrialFunction(V), TestFunction(V)
        uh = Function(V)
        solve((inner(curl(u), curl(v)) + inner(u, v)) * dx == inner(f, v) * dx, uh, DirichletBC(V, uex, "on_boundary"))
        errs.append(errornorm(uex, uh))
        hs.append(1.0 / n)
    assert _rates(errs, hs)[0] > 1.7


def test_dg_sipg_poisson_convergence():
    errs, hs = [], []
    for n in (4, 8):
        m = UnitSquareMesh(n, n)
        V = FunctionSpace(m, "DG", 2)
        u, v = TrialFunction(V), TestFunction(V)
        nn, h = FacetNormal(m), CellDiameter(m)
        x = SpatialCoordinate(m)
        uex = sin(pi * x[0]) * sin(pi * x[1])
        al = 10.0
        a = (inner(grad(u), grad(v)) * dx - inner(avg(grad(u)), jump(v, nn)) * dS - inner(jump(u, nn), avg(grad(v))) * dS
             + al / avg(h) * inner(jump(u, nn), jump(v, nn)) * dS
             - inner(grad(u), nn) * v * ds - u * inner(grad(v), nn) * ds + al / h * u * v * ds)
        uh = Function(V)
        solve(a == 2 * pi ** 2 * uex * v * dx, uh)
        errs.append(errornorm(uex, uh))
        hs.append(1.0 / n)
    assert _rates(errs, hs)[0] > 2.6


def test_biharmonic_c0ip():
    m = UnitSquareMesh(8, 8)
    V = FunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    h = CellDiameter(m)
    n = FacetNormal(m)
    h_avg = (h("+") + h("-")) / 2.0
    x = SpatialCoordinate(m)
    uex = sin(pi * x[0]) * sin(pi * x[1])
    f = 4.0 * pi ** 4 * uex
    a = (inner(div(grad(u)), div(grad(v))) * dx - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
         - inner(jump(grad(u), n), avg(div(grad(v)))) * dS + 8.0 / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS)
    uh = Function(V)
    solve(a == f * v * dx, uh, DirichletBC(V, 0.0, "on_boundary"))
    assert errornorm(uex, uh) < 0.05


# ---------- Newton

def _nl_problem(m, k=2):
    V = FunctionSpace(m, "Lagrange", k)
    x = SpatialCoordinate(m)
    uex = 1 + x[0] + 2 * x[1]
    q = lambda w: 1 + w ** 2
    f = -div(q(uex) * grad(uex))
    return V, uex, q, f


def test_newton_ad_and_ufl_jacobian_agree(sq):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    bc = DirichletBC(V, uex, "on_boundary")
    u1, u2 = Function(V), Function(V)
    F1 = q(u1) * dot(grad(u1), grad(v)) * dx - f * v * dx
    F2 = q(u2) * dot(grad(u2), grad(v)) * dx - f * v * dx
    solve(F1 == 0, u1, bc)
    solve(F2 == 0, u2, bc, J=derivative(F2, u2))
    assert errornorm(uex, u1) < 1e-9
    assert errornorm(uex, u2) < 1e-9
    assert np.abs(u1.array - u2.array).max() < 1e-9


def test_newton_equation_with_rhs_form(sq):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    solve(q(u) * dot(grad(u), grad(v)) * dx == f * v * dx, u, DirichletBC(V, uex, "on_boundary"))
    assert errornorm(uex, u) < 1e-9


def test_newton_iterations_and_monitor(sq, capsys):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    s = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, DirichletBC(V, uex, "on_boundary")),
                                   {"snes_monitor": True})
    s.solve()
    out = capsys.readouterr().out
    assert "Newton iteration 0" in out
    assert 2 <= s.iterations <= 10
    s.solve()
    assert s.iterations <= 1


def test_newton_quadratic_convergence(sq, capsys):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    solve(F == 0, u, DirichletBC(V, uex, "on_boundary"),
          solver_parameters={"snes_monitor": True, "snes_atol": 1e-14, "snes_rtol": 1e-14})
    res = [float(l.split("=")[1]) for l in capsys.readouterr().out.splitlines() if "residual" in l]
    assert res[-1] < 1e-10
    for a, b in zip(res[1:], res[2:]):
        if a > 1e-8:
            assert b < 10 * a ** 2 * 1e2 + 1e-13


def test_newton_max_it_raises(sq):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    with pytest.raises(RuntimeError):
        solve(F == 0, u, DirichletBC(V, uex, "on_boundary"), solver_parameters={"snes_max_it": 1, "snes_rtol": 1e-14, "snes_atol": 1e-14})


def test_newton_damping_parameter(sq):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    s = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, DirichletBC(V, uex, "on_boundary")),
                                   {"snes_damping": 0.5, "snes_max_it": 200})
    s.solve()
    assert errornorm(uex, u) < 1e-8
    u2 = Function(V)
    F2 = q(u2) * dot(grad(u2), grad(v)) * dx - f * v * dx
    s2 = NonlinearVariationalSolver(NonlinearVariationalProblem(F2, u2, DirichletBC(V, uex, "on_boundary")))
    s2.solve()
    assert s.iterations > s2.iterations


def test_newton_krylov_inner_solver(sq):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    solve(F == 0, u, DirichletBC(V, uex, "on_boundary"),
          solver_parameters={"ksp_type": "gmres", "pc_type": "jacobi", "ksp_rtol": 1e-12})
    assert errornorm(uex, u) < 1e-8


def test_newton_vector_hyperelastic_ad_vs_ufl():
    m = BoxMesh(3, 2, 2, 1.5, 1.0, 1.0)
    V = VectorFunctionSpace(m, "Lagrange", 1)
    v, du = TestFunction(V), TrialFunction(V)
    mu, lmbda = 1.0, 1.5
    B = Constant(m, (0.0, -0.3, 0.0))
    bc = DirichletBC(V, (0, 0, 0), "back")
    sols = []
    for use_j in (False, True):
        u = Function(V)
        Fd = Identity(3) + grad(u)
        psi = (mu / 2) * (tr(Fd.T * Fd) - 3) - mu * ln(det(Fd)) + (lmbda / 2) * ln(det(Fd)) ** 2
        Fres = derivative(psi * dx - dot(B, u) * dx, u, v)
        solve(Fres == 0, u, bc, J=derivative(Fres, u, du) if use_j else None)
        sols.append(u.array.copy())
    assert np.abs(sols[0] - sols[1]).max() < 1e-8
    assert np.abs(sols[0]).max() > 1e-3


def test_newton_mixed_system_cahn_hilliard_mass():
    m = UnitSquareMesh(6, 6)
    P1 = FunctionSpace(m, "Lagrange", 1)
    ME = P1 * P1
    q, v = TestFunctions(ME)
    u, u0 = Function(ME), Function(ME)
    c, mu = split(u)
    c0, mu0 = split(u0)
    rng = np.random.default_rng(3)
    u.sub(0).array[:] = 0.63 + 0.02 * (0.5 - rng.random(P1.dim()))
    u0.assign(u)
    c = variable(c)
    f = 100 * c ** 2 * (1 - c) ** 2
    dfdc = diff(f, c)
    dt = 5e-6
    F = c * q * dx - c0 * q * dx + dt * dot(grad(0.5 * mu0 + 0.5 * mu), grad(q)) * dx \
        + mu * v * dx - dfdc * v * dx - 1e-2 * dot(grad(c), grad(v)) * dx
    m0 = assemble(u.sub(0) * dx)
    solve(F == 0, u, solver_parameters={"snes_rtol": 1e-10})
    assert assemble(u.sub(0) * dx) == approx(m0, rel=1e-9)
    assert np.isfinite(u.array).all()


def test_newton_with_dirichlet_on_vector_subspace_and_update_between_solves(sq):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    k = Constant(sq, 1.0)
    F = k * q(u) * dot(grad(u), grad(v)) * dx - k * f * v * dx
    bc = DirichletBC(V, uex, "on_boundary")
    s = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, bc))
    s.solve()
    e1 = errornorm(uex, u)
    k.assign(3.0)
    u.interpolate(0.0)
    s.solve()
    assert e1 < 1e-9 and errornorm(uex, u) < 1e-9


def test_newton_dg_nonlinear_facet_terms_with_ufl_jacobian():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "DG", 1)
    v = TestFunction(V)
    u = Function(V).interpolate(0.3)
    n = FacetNormal(m)
    c = Constant(m, (1.0, 0.5))
    F = (u ** 3 * v + 0.1 * dot(grad(u), grad(v)) + 0.5 * u * v) * dx + 0.2 * jump(u) ** 3 * jump(v) * dS \
        + dot(avg(u) * c, n("+")) * jump(v) * dS - Constant(m, 1.0) * v * dx
    solve(F == 0, u, J=derivative(F, u))
    r = assemble(F).FV().NumPy()
    assert np.abs(r).max() < 1e-8


def test_newton_dg_nonlinear_facet_terms_ngs_jacobian():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "DG", 1)
    v = TestFunction(V)
    u = Function(V).interpolate(lambda x: 0.3 + 0.2 * x[0] * x[1])
    F = (u ** 3 * v + 0.5 * u * v) * dx + 0.2 * avg(u) ** 2 * jump(v) * dS - Constant(m, 1.0) * v * dx
    solve(F == 0, u)
    assert np.abs(assemble(F).FV().NumPy()).max() < 1e-8


def test_nonlinear_problem_objects_reusable(sq):
    V, uex, q, f = _nl_problem(sq)
    v = TestFunction(V)
    u = Function(V)
    F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx
    prob = NonlinearVariationalProblem(F, u, DirichletBC(V, uex, "on_boundary"), J=derivative(F, u))
    NonlinearVariationalSolver(prob).solve()
    assert errornorm(uex, u) < 1e-9


# ---------- LinearVariationalSolver / time loop

def test_heat_time_loop_exact_in_time():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 2)
    x = SpatialCoordinate(m)
    t, dt = Constant(m, 0.0), 0.1
    uex = 1 + x[0] ** 2 + 3 * x[1] ** 2 + 1.2 * t
    u, v = TrialFunction(V), TestFunction(V)
    un = Function(V).interpolate(uex)
    F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (un + dt * Constant(m, 1.2 - 2 - 6)) * v * dx
    a, L = lhs(F), rhs(F)
    uh = Function(V)
    bc = DirichletBC(V, uex, "on_boundary")
    for cj in (True, False):
        t.assign(0.0)
        un.interpolate(uex)
        s = LinearVariationalSolver(LinearVariationalProblem(a, L, uh, bc, constant_jacobian=cj))
        for _ in range(5):
            t.assign(t.value + dt)
            s.solve()
            un.assign(uh)
        assert errornorm(uex, uh) < 1e-11, cj


def test_constant_jacobian_semantics():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 1)
    c = Constant(m, 1.0)
    u, v = TrialFunction(V), TestFunction(V)
    a = c * u * v * dx
    L = Constant(m, 1.0) * v * dx
    uh = Function(V)
    bcs = None
    s_const = LinearVariationalSolver(LinearVariationalProblem(a, L, uh, bcs, constant_jacobian=True))
    s_upd = LinearVariationalSolver(LinearVariationalProblem(a, L, uh, bcs, constant_jacobian=False))
    s_const.solve()
    assert uh.at(0.5, 0.5) == approx(1.0)
    c.assign(2.0)
    s_upd.solve()
    assert uh.at(0.5, 0.5) == approx(0.5)
    s_const.solve()
    assert uh.at(0.5, 0.5) == approx(1.0)


def test_linear_variational_rhs_updates_each_solve():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    s_ = Constant(m, 1.0)
    uh = Function(V)
    solver = LinearVariationalSolver(LinearVariationalProblem(u * v * dx, s_ * v * dx, uh, constant_jacobian=True))
    for val in (1.0, 2.0, -3.0):
        s_.assign(val)
        solver.solve()
        assert uh.at(0.5, 0.5) == approx(val)


def test_linear_variational_with_krylov_and_bc_update():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    g = Constant(m, 1.0)
    uh = Function(V)
    solver = LinearVariationalSolver(
        LinearVariationalProblem(inner(grad(u), grad(v)) * dx, Constant(m, 0.0) * v * dx, uh, DirichletBC(V, g, "on_boundary"),
                                 constant_jacobian=True),
        {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12})
    solver.solve()
    assert uh.at(0.5, 0.5) == approx(1.0)
    g.assign(4.0)
    solver.solve()
    assert uh.at(0.5, 0.5) == approx(4.0)


def test_linear_variational_problem_zero_rhs_none():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V).interpolate(1.0)
    LinearVariationalSolver(LinearVariationalProblem(u * v * dx, None, uh, DirichletBC(V, 0.0, "on_boundary"))).solve()
    assert np.abs(uh.array).max() < 1e-12


def test_assemble_flags_and_dg_linear_solve_direct_vs_cg():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "DG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(m)
    a = u * v * dx + 0.1 * jump(u) * jump(v) * dS
    L = Constant(m, 1.0) * v * dx
    s1, s2 = Function(V), Function(V)
    solve(a == L, s1)
    solve(a == L, s2, solver_parameters={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-13})
    assert np.abs(s1.array - s2.array).max() < 1e-9
    assert s1.at(0.4, 0.4) == approx(1.0)


# ---------- project / interpolate / norms

def test_project_exact_and_dg0():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 2)
    x = SpatialCoordinate(m)
    w = project(x[0] * x[1] + x[0] ** 2, V)
    assert errornorm(x[0] * x[1] + x[0] ** 2, w) < 1e-12
    Q = FunctionSpace(m, "DG", 0)
    p0 = project(x[0], Q)
    ref = ngs.GridFunction(ngs.L2(m.ngs, order=0))
    ref.Set(ngs.x)
    for pt in [(0.1, 0.1), (0.8, 0.3), (0.45, 0.9)]:
        assert p0.at(*pt) == approx(ref(m.ngs(*pt)), abs=1e-12)
    assert assemble(p0 * dx(domain=m)) == approx(0.5)


def test_project_vector_function_callable_and_number():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    w = project(as_vector((SpatialCoordinate(m)[1], 1.0)), V)
    assert w.at(0.3, 0.4) == approx([0.4, 1.0])
    w = project(lambda x: (x[0], x[1] ** 2), V)
    assert w.at(0.3, 0.4) == approx([0.3, 0.16])
    Q = FunctionSpace(m, "Lagrange", 1)
    assert project(2.0, Q).at(0.3, 0.3) == approx(2.0)


def test_project_with_bcs_and_name():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 1)
    w = project(Constant(m, 2.0), V, bcs=DirichletBC(V, 5.0, "left"), name="pw")
    assert w.name == "pw"
    assert w.at(0.0, 0.5) == approx(5.0)


def test_project_function_into_other_space():
    m = UnitSquareMesh(4, 4)
    V3 = FunctionSpace(m, "Lagrange", 3)
    V1 = FunctionSpace(m, "Lagrange", 1)
    f = Function(V1).interpolate(lambda x: 1 + 2 * x[0])
    g = project(f, V3)
    assert errornorm(f, g) < 1e-12


def test_interpolate_function_free():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 2)
    f = interpolate(SpatialCoordinate(m)[0] ** 2, V, name="ff")
    assert f.name == "ff"
    assert f.at(0.3, 0.1) == approx(0.09)
    g = interpolate(lambda x: x[1] + 1, V)
    assert g.at(0.3, 0.1) == approx(1.1)


def test_norms_expressions():
    m = UnitSquareMesh(6, 6)
    x = SpatialCoordinate(m)
    assert norm(x[0]) == approx(1 / math.sqrt(3), rel=1e-12)
    assert norm(x[0], "L2") == approx(1 / math.sqrt(3), rel=1e-12)
    assert norm(x[0], "H10") == approx(1.0)
    assert norm(x[0], "h1") == approx(math.sqrt(4 / 3), rel=1e-12)
    assert norm(Constant(m, 3.0)) == approx(3.0)
    assert norm(x) == approx(math.sqrt(2 / 3), rel=1e-12)
    assert norm(as_vector((-x[1], x[0])), "Hcurl") == approx(math.sqrt(2 / 3 + 4), rel=1e-12)
    assert norm(x, "Hdiv") == approx(math.sqrt(2 / 3 + 4), rel=1e-12)
    assert norm(x[0] ** 2, "H10") == approx(math.sqrt(4 / 3), rel=1e-12)
    assert norm(x[0], "L4") == approx(0.2 ** 0.25, rel=1e-12)
    with pytest.raises(ValueError):
        norm(x[0], "W7")


def test_norm_of_function_and_vector_function():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] * x[1])
    assert norm(f) == approx(1 / 3, rel=1e-12)
    assert norm(f, "H10") == approx(math.sqrt(2 / 3), rel=1e-12)
    W = VectorFunctionSpace(m, "Lagrange", 2)
    w = Function(W).interpolate(lambda x: (x[0], x[1]))
    assert norm(w) == approx(math.sqrt(2 / 3), rel=1e-12)
    assert norm(w, "H1") == approx(math.sqrt(2 / 3 + 2), rel=1e-12)
    assert norm(w, "Hdiv") == approx(math.sqrt(2 / 3 + 4), rel=1e-12)


def test_errornorm_function_vs_expression():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 1)
    x = SpatialCoordinate(m)
    f = Function(V).interpolate(lambda x: x[0])
    assert errornorm(x[0], f) < 1e-14
    assert errornorm(x[0] + 1, f) == approx(1.0)
    assert errornorm(x[0] + 1, f, "H1") == approx(1.0)
    assert errornorm(x[0] ** 2, f, "H10") > 0.1
    g = Function(V).interpolate(lambda x: x[0] + 0.5)
    assert errornorm(f, g) == approx(0.5)
    assert errornorm(g, f) == approx(0.5)


def test_errornorm_on_mixed_subfunction_and_vector():
    m = UnitSquareMesh(4, 4)
    W = VectorFunctionSpace(m, "Lagrange", 1) * FunctionSpace(m, "Lagrange", 1)
    w = Function(W)
    w.sub(0).interpolate((1.0, 2.0))
    w.sub(1).interpolate(3.0)
    u, p = w.subfunctions
    assert errornorm(as_vector((1.0, 2.0)), u) < 1e-14
    assert errornorm(Constant(m, 3.0), p) < 1e-14
    uu, pp = split(w)
    assert errornorm(as_vector((1.0, 2.0)), uu) < 1e-14


# ---------- output

@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg")
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_2d_function_expression_vector_mesh(tmp_path):
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 2)
    W = VectorFunctionSpace(m, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] * x[1])
    w = Function(W).interpolate(lambda x: (x[0], 2 * x[1]))
    x = SpatialCoordinate(m)
    ax = plot(f)
    assert ax is not None
    ax.figure.savefig(tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 1000
    assert plot(w, colorbar=False) is not None
    assert plot(sin(x[0])) is not None
    assert plot(grad(f)) is not None
    ax2 = plot(m)
    assert plot(f, ax=ax2, subdivision=2) is ax2
    assert plot(f, subdivision=1, cmap="plasma") is not None


def test_plot_values_correct_on_vertices():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: 1 + x[0])
    ax = plot(f, subdivision=1)
    coll = ax.collections[0]
    arr = np.asarray(coll.get_array())
    assert arr.min() == approx(1.0) and arr.max() == approx(2.0)


def test_plot_quads_and_dg():
    m = UnitSquareMesh(3, 3, quadrilateral=True)
    V = FunctionSpace(m, "DG", 1)
    f = Function(V).interpolate(lambda x: x[0])
    assert plot(f) is not None
    assert plot(m) is not None


def test_plot_hdiv_function():
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, "RT", 1)
    f = Function(V).interpolate(lambda x: (x[0], x[1]))
    assert plot(f) is not None


def test_plot_1d():
    m = UnitIntervalMesh(6)
    V = FunctionSpace(m, "Lagrange", 2)
    f = Function(V).interpolate(lambda x: x[0] ** 2)
    ax = plot(f)
    line = ax.lines[0]
    xs, ys = line.get_data()
    assert np.all(np.diff(xs) >= -1e-14)
    assert np.allclose(ys, xs ** 2, atol=1e-12)
    assert plot(SpatialCoordinate(m)[0] ** 2) is not None


def test_plot_3d_with_deformation():
    m = BoxMesh(2, 2, 2)
    V = FunctionSpace(m, "Lagrange", 1)
    W = VectorFunctionSpace(m, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: x[0] + x[2])
    d = Function(W).interpolate(lambda x: (0.0 * x[0], 0.0 * x[0], 0.2 * x[0]))
    ax = plot(f)
    assert ax is not None
    ax2 = plot(f, deformation=d)
    assert ax2.get_zlim()[1] > 1.1
    ax3 = plot(f, deformation=as_vector((0.0, 0.0, 0.2 * SpatialCoordinate(m)[0])), colorbar=False)
    assert ax3 is not None
    assert plot(Function(W).interpolate((1.0, 0.0, 0.0))) is not None


def test_plot_3d_hex():
    m = BoxMesh(2, 2, 2, hexahedral=True)
    V = FunctionSpace(m, "Lagrange", 1)
    f = Function(V).interpolate(lambda x: x[0])
    assert plot(f) is not None


def test_plot_mesh_3d_not_supported():
    with pytest.raises(NotImplementedError):
        plot(BoxMesh(1, 1, 1))


def test_plot_mixed_subfunction():
    m = UnitSquareMesh(3, 3)
    W = VectorFunctionSpace(m, "Lagrange", 2) * FunctionSpace(m, "Lagrange", 1)
    w = Function(W)
    u, p = w.subfunctions
    p.interpolate(lambda x: x[0])
    assert plot(p) is not None
    assert plot(u) is not None


def test_vtkfile_write(tmp_path):
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, "Lagrange", 1)
    W = VectorFunctionSpace(m, "Lagrange", 1)
    f, w = Function(V, name="f"), Function(W, name="w")
    f.interpolate(lambda x: x[0])
    out = VTKFile(str(tmp_path / "out"))
    out.write(f, w)
    out.write(f, w, time=0.1)
    files = os.listdir(tmp_path)
    assert any(n.startswith("out") for n in files)
    assert any(n.endswith(".vtu") for n in files)


def test_vtkfile_extension_stripped(tmp_path):
    m = UnitSquareMesh(2, 2)
    f = Function(FunctionSpace(m, "Lagrange", 1), name="f")
    VTKFile(str(tmp_path / "x.pvd")).write(f, time=0.0)
    assert any(n.startswith("x") for n in os.listdir(tmp_path))


# ---------- more solver scenarios

def test_navier_stokes_manufactured_newton_with_real_multiplier():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    R = FunctionSpace(m, "Real", 0)
    W = V * Q * R
    x = SpatialCoordinate(m)
    nu = 0.5
    uex = as_vector((x[0] ** 2, -2 * x[0] * x[1]))
    pex = x[0] + x[1] - 1
    f = -nu * div(grad(uex)) + grad(uex) * uex + grad(pex)
    sols = []
    for use_j in (False, True):
        w = Function(W)
        u, p, c = split(w)
        v, q, r = TestFunctions(W)
        F = (nu * inner(grad(u), grad(v)) + dot(grad(u) * u, v) - p * div(v) - q * div(u) + c * q + r * p) * dx - dot(f, v) * dx
        J = derivative(F, w, TrialFunction(W)) if use_j else None
        solve(F == 0, w, DirichletBC(W.sub(0), uex, "on_boundary"), J=J)
        sols.append(w.array.copy())
        uh, ph, ch = w.subfunctions
        assert errornorm(uex, uh) < 1e-9
        assert errornorm(pex, ph) < 1e-9
    assert np.abs(sols[0] - sols[1]).max() < 1e-8


def test_bc_g_function_from_collapsed_subspace():
    m = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    Q = FunctionSpace(m, "Lagrange", 1)
    W = V * Q
    g = Function(V).interpolate(lambda x: (1 + x[0], 2 * x[1]))
    bc = DirichletBC(W.sub(0), g, "on_boundary")
    w = Function(W)
    bc.apply(w.gf)
    u, p = w.subfunctions
    assert u.at(1.0, 0.5) == approx([2.0, 1.0])
    assert u.at(0.3, 0.0) == approx([1.3, 0.0])


def test_bc_on_rt_space_normal_flux():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "RT", 2)
    g = Function(V).interpolate(lambda x: (x[0] + 2 * x[1], 3 * x[0] - x[1]))
    w = Function(V)
    DirichletBC(V, g, "on_boundary").apply(w.gf)
    n = FacetNormal(m)
    assert assemble(dot(w - g, n) ** 2 * ds) < 1e-24


def test_bc_on_hcurl_tangential():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "N1curl", 2)
    ge = lambda x: (x[1] + x[0] * x[1], x[0] - x[1] ** 2)
    g = Function(V).interpolate(ge)
    w = Function(V)
    DirichletBC(V, g, "bottom").apply(w.gf)
    n = FacetNormal(m)
    assert assemble(dot(w - g, perp(n)) ** 2 * ds("bottom")) < 1e-24
    assert norm(w) > 0


@pytest.mark.parametrize("hexa", [False, True])
def test_bc_callable_3d(hexa):
    m = BoxMesh(2, 2, 2, hexahedral=hexa)
    V = FunctionSpace(m, "Lagrange", 2)
    x = SpatialCoordinate(m)
    bc = DirichletBC(V, 1 + x[2], lambda x: isc(x[2], 0.0) | isc(x[2], 1.0))
    assert sum(bc.dofs()) == sum(DirichletBC(V, 0.0, "top|bottom").dofs())
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    solve(inner(grad(u), grad(v)) * dx == Constant(m, 0.0) * v * dx, uh, bc)
    assert errornorm(1 + x[2], uh) < 1e-12


def test_bc_in_linear_variational_solver_changes_g_function():
    m = UnitSquareMesh(4, 4)
    V = FunctionSpace(m, "Lagrange", 1)
    g = Function(V).interpolate(1.0)
    u, v = TrialFunction(V), TestFunction(V)
    uh = Function(V)
    s = LinearVariationalSolver(LinearVariationalProblem(inner(grad(u), grad(v)) * dx, Constant(m, 0.0) * v * dx, uh,
                                                         DirichletBC(V, g, "on_boundary"), constant_jacobian=True))
    s.solve()
    assert uh.at(0.5, 0.5) == approx(1.0)
    g.interpolate(2.0)
    s.solve()
    assert uh.at(0.5, 0.5) == approx(2.0)


def test_elasticity_cantilever_vs_native():
    m = BoxMesh(6, 2, 2, 3.0, 1.0, 1.0)
    V = VectorFunctionSpace(m, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    mu, lam = 1.0, 1.25
    eps = lambda w: sym(grad(w))
    sig = lambda w: lam * tr(eps(w)) * Identity(3) + 2 * mu * eps(w)
    L = dot(Constant(m, (0.0, 0.0, -0.01)), v) * dx + dot(Constant(m, (0, 0, -0.001)), v) * ds("front")
    uh = Function(V)
    solve(inner(sig(u), eps(v)) * dx == L, uh, DirichletBC(V, (0, 0, 0), "back"))
    fes = ngs.VectorH1(m.ngs, order=2, dirichlet="back")
    U, W = fes.TnT()
    E = lambda w: ngs.Sym(ngs.Grad(w))
    a = ngs.BilinearForm(ngs.InnerProduct(lam * ngs.Trace(E(U)) * ngs.Id(3) + 2 * mu * E(U), E(W)) * ngs.dx).Assemble()
    f = ngs.LinearForm(ngs.CF((0, 0, -0.01)) * W * ngs.dx + ngs.CF((0, 0, -0.001)) * W * ngs.ds("front")).Assemble()
    g = ngs.GridFunction(fes)
    g.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
    pt = (3.0, 0.5, 0.5)
    assert uh.at(*pt) == approx(np.array(g(m.ngs(*pt))), rel=1e-9)
    assert np.abs(uh.array - g.vec.FV().NumPy()).max() < 1e-9


def test_plot_and_vtk_of_mixed_function(tmp_path):
    m = UnitSquareMesh(3, 3)
    w = Function(VectorFunctionSpace(m, "Lagrange", 2) * FunctionSpace(m, "Lagrange", 1))
    plot(w)
    VTKFile(str(tmp_path / "mix")).write(w)


def test_plot_unsupported_kwargs_pass_through_to_matplotlib():
    m = UnitSquareMesh(3, 3)
    f = Function(FunctionSpace(m, "Lagrange", 1)).interpolate(lambda x: x[0])
    ax = plot(f, vmin=-1.0, vmax=2.0)
    assert ax.collections[0].get_clim() == (-1.0, 2.0)


def test_constant_without_mesh_is_domain_independent():
    ma = RectangleMesh(2, 2, 2.0, 1.0)
    mb = UnitSquareMesh(3, 3)
    c = Constant(2.0)
    assert assemble(c * dx(domain=ma)) == approx(4.0)
    assert assemble(c * dx(domain=mb)) == approx(2.0)


@pytest.mark.xfail(reason="coefficients/coordinates of another mesh are silently integrated on the measure's mesh", strict=True)
def test_mixing_meshes_in_one_integral_is_detected():
    ma = RectangleMesh(2, 2, 2.0, 1.0)
    mb = UnitSquareMesh(3, 3)
    with pytest.raises(Exception):
        assemble(SpatialCoordinate(ma)[0] * dx(domain=mb))

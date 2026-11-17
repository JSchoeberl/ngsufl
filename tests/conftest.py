import numpy as np
import pytest

from ngsufl import (BoxMesh, Constant, Function, FunctionSpace, UnitIntervalMesh,
                    UnitSquareMesh, VectorFunctionSpace)

P2D = [(0.3, 0.4), (0.7, 0.2), (0.55, 0.85)]
P3D = [(0.3, 0.4, 0.2), (0.7, 0.2, 0.6), (0.15, 0.85, 0.45)]
P1D = [(0.3,), (0.8,)]


@pytest.fixture(scope="session")
def mesh2():
    return UnitSquareMesh(4, 4)


@pytest.fixture(scope="session")
def mesh3():
    return BoxMesh(2, 2, 2)


@pytest.fixture(scope="session")
def mesh1():
    return UnitIntervalMesh(6)


@pytest.fixture(scope="session")
def meshq():
    return UnitSquareMesh(4, 4, quadrilateral=True)


def ev(cf, mesh, p):
    """Evaluate a CF (or number) at a point, as numpy array."""
    if isinstance(cf, (int, float)):
        return np.array(cf)
    v = cf(mesh.ngs(*p))
    return np.array(v, dtype=float)


def evshape(expr, mesh, p, to_cf):
    r = ev(to_cf(expr), mesh, p)
    return r.reshape(expr.ufl_shape) if r.size == int(np.prod(expr.ufl_shape, dtype=int)) else r


def approx(x, **kw):
    return pytest.approx(np.asarray(x, dtype=float), **kw)


import ngsolve as ngs


def dense(mat):
    return mat.ToDense().NumPy().copy()


def native_bf(fes, expr_fn, trial=None, **kw):
    """Dense matrix of a native BilinearForm; expr_fn(U, W) -> integral expression."""
    if trial is None:
        U, W = fes.TnT()
        a = ngs.BilinearForm(fes, **kw)
    else:
        U, W = trial.TrialFunction(), fes.TestFunction()
        a = ngs.BilinearForm(trialspace=trial, testspace=fes, **kw)
    a += expr_fn(U, W)
    a.Assemble()
    return dense(a.mat)


def native_lf(fes, expr_fn):
    f = ngs.LinearForm(fes)
    f += expr_fn(fes.TestFunction())
    f.Assemble()
    return f.vec.FV().NumPy().copy()


def close(a, b, tol=1e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    scale = max(np.abs(b).max(), 1.0)
    assert np.abs(a - b).max() <= tol * scale

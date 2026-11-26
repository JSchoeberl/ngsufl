"""Legacy FEniCS (dolfin 2019) front end: ``from ngsufl.fenics import *``.

Classic scripts run after replacing ``from fenics import *`` by this import.

Differences worth knowing:

* Meshes are structured NGSolve meshes (``UnitSquareMesh``, ``RectangleMesh``, ...),
  no mesh files, ``MeshFunction`` or ``refine``.
* ``Expression`` parses the C++ expression strings (no ``UserExpression``).  It belongs
  to the mesh that was created last; parameters are updated by attribute.
* ``DirichletBC`` takes ``on_boundary``, C++ condition strings, ``SubDomain`` objects and
  functions ``(x, on_boundary)``.  A facet is constrained if all its vertices are marked.
  ``SubDomain.mark`` is not available.
* Solver names are mapped to the PETSc style options of the core: ``lu``, ``default``,
  ``mumps``, ``umfpack``, ``superlu``, ``petsc`` -> direct solver; ``cg``, ``gmres``,
  ``minres`` -> that Krylov method (``bicgstab``, ``tfqmr`` -> gmres).  Preconditioners:
  ``none`` -> none, ``lu`` -> exact, everything else (``ilu``, ``icc``, ``sor``, ``amg``,
  ``hypre_amg``, ``default``) -> Jacobi, because NGSolve has no ILU here.
  Krylov tolerances default to 1e-10 (relative).
* Coefficient vectors belong to NGSolve's hierarchical basis, they are point values only
  for P1/P0.  Use ``compute_vertex_values`` for vertex values.
"""

import math
import numbers
import os
import re

import ngsolve as ngs
import numpy as np
import ufl
import ngsufl as _core
from ngsufl import *  # noqa: F401,F403
from ngsufl import __all__ as _core_all
from ngsufl import mesh as _meshmod
from ngsufl import output as _output
from ngsufl import pyplot as _pyplot
from ngsufl import solving as _solving
from ngsufl.bcs import DirichletBC as _CoreBC
from ngsufl.function import ExternalFunction, Vector
from ngsufl.function import Function as _CoreFunction
from ngsufl.function import FunctionSpace as _CoreFunctionSpace
from ngsufl.mesh import Mesh as _CoreMesh
from ngsufl.mesh import default_mesh
from ngsufl.solving import ConvergenceError, Matrix

DOLFIN_EPS = 3.0e-16
DOLFIN_PI = math.pi

# ---------------------------------------------------------------- helpers


def near(x, x0, eps=DOLFIN_EPS):
    return x0 - eps <= x <= x0 + eps


def between(x, interval):
    return interval[0] - DOLFIN_EPS <= x <= interval[1] + DOLFIN_EPS


class _Params(dict):
    """Nested dict that creates sub-dictionaries on access, also usable with attributes."""

    def __missing__(self, key):
        v = self[key] = _Params()
        return v

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def _nested(d):
    out = _Params()
    for k, v in d.items():
        out[k] = _nested(v) if isinstance(v, dict) else v
    return out


parameters = _Params({"form_compiler": {"cpp_optimize": True, "quadrature_degree": "auto"},
                      "allow_extrapolation": False, "reorder_dofs_serial": True,
                      "linear_algebra_backend": "PETSc"})

_log_level = 20


class LogLevel:
    CRITICAL, ERROR, WARNING, INFO, PROGRESS, TRACE, DEBUG = 50, 40, 30, 20, 16, 13, 10


def set_log_level(level):
    global _log_level
    _log_level = int(level)


def set_log_active(active=True):
    global _log_level
    _log_level = 20 if active else 1000


def info(obj, *args):
    if _log_level <= LogLevel.INFO:
        print(obj if isinstance(obj, str) else str(obj))


def interactive(*args, **kwargs):
    pass


class _MPI:
    comm_world = None

    @staticmethod
    def rank(comm=None):
        return 0

    @staticmethod
    def size(comm=None):
        return 1


MPI = _MPI()


def mpi_comm_world():
    return None


# ---------------------------------------------------------------- unsupported names

_UNSUPPORTED = {
    "MeshFunction": "mesh functions", "MeshFunctionSizet": "mesh functions",
    "MeshFunctionInt": "mesh functions", "MeshFunctionDouble": "mesh functions",
    "MeshFunctionBool": "mesh functions", "FacetFunction": "mesh functions",
    "CellFunction": "mesh functions", "VertexFunction": "mesh functions",
    "MeshValueCollection": "mesh value collections", "XDMFFile": "XDMF files",
    "HDF5File": "HDF5 files", "TimeSeries": "time series", "UserExpression": "UserExpression",
    "refine": "mesh refinement", "adapt": "adaptive refinement", "SubMesh": "sub meshes",
    "MeshEditor": "mesh editing", "vertex_to_dof_map": "vertex_to_dof_map",
    "dof_to_vertex_map": "dof_to_vertex_map", "FunctionAssigner": "FunctionAssigner",
    "PeriodicBoundary": "periodic boundaries", "NonlinearProblem": "NonlinearProblem",
    "NewtonSolver": "NewtonSolver", "PETScKrylovSolver": "PETSc solver objects",
    "PETScLUSolver": "PETSc solver objects", "Cell": "cell iteration", "Facet": "facet iteration",
    "cells": "cell iteration", "facets": "facet iteration", "vertices": "vertex iteration",
}


class _Unsupported:
    _what = ""

    @classmethod
    def _fail(cls):
        base = next(c for c in cls.__mro__ if c.__dict__.get("_what"))
        raise NotImplementedError(f"'{base.__name__}' is not available in ngsufl.fenics yet "
                                  f"({base._what})")

    def __init_subclass__(cls, **kw):
        if "_what" not in cls.__dict__:
            cls._fail()

    def __new__(cls, *args, **kwargs):
        cls._fail()


def _stub(name, what):
    return type(name, (_Unsupported,), {"_what": what})


_STUBS = {n: _stub(n, w) for n, w in _UNSUPPORTED.items()}
globals().update(_STUBS)


class _Mshr:
    def __getattr__(self, name):
        raise NotImplementedError("'mshr' is not available in ngsufl.fenics yet")


mshr = _Mshr()
_STUBS["mshr"] = mshr


def __getattr__(name):
    if name in _UNSUPPORTED or name.startswith("mshr"):
        raise NotImplementedError(f"'{name}' is not available in ngsufl.fenics yet")
    raise AttributeError(f"module 'ngsufl.fenics' has no attribute '{name}'")


# ---------------------------------------------------------------- expression language

class ExpressionError(ValueError):
    pass


_NUM = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_TOKEN = re.compile(rf"\s*(?:({_NUM})|([A-Za-z_]\w*)|(&&|\|\||<=|>=|==|!=|[-+*/<>!?:()\[\],]))")

_CONSTS = {"pi": math.pi, "DOLFIN_PI": math.pi, "M_PI": math.pi, "DOLFIN_EPS": DOLFIN_EPS,
           "M_E": math.e}
_ARITY = {"pow": (2, 2), "sqrt": (1, 1), "exp": (1, 1), "log": (1, 1), "sin": (1, 1),
          "cos": (1, 1), "tan": (1, 1), "asin": (1, 1), "acos": (1, 1), "atan": (1, 1),
          "atan2": (2, 2), "sinh": (1, 1), "cosh": (1, 1), "tanh": (1, 1), "fabs": (1, 1),
          "abs": (1, 1), "min": (2, 2), "max": (2, 2), "fmin": (2, 2), "fmax": (2, 2),
          "near": (2, 3)}


def _tokenize(src):
    s = src.rstrip()
    if not s.strip():
        raise ExpressionError("empty expression")
    pos, out = 0, []
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ExpressionError(f"unexpected character {s[pos:].lstrip()[0]!r} in '{src}'")
        pos = m.end()
        if m.group(1) is not None:
            out.append(("num", m.group(1)))
        elif m.group(2) is not None:
            out.append(("id", m.group(2)))
        else:
            out.append(("op", m.group(3)))
    return out


class _Parser:
    def __init__(self, src):
        self.src, self.tok, self.i = src, _tokenize(src), 0

    def fail(self, msg):
        raise ExpressionError(f"{msg} in '{self.src}'")

    def peek(self):
        return self.tok[self.i] if self.i < len(self.tok) else (None, None)

    def accept(self, *ops):
        k, t = self.peek()
        if k == "op" and t in ops:
            self.i += 1
            return t
        return None

    def expect(self, op):
        if not self.accept(op):
            k, t = self.peek()
            self.fail(f"expected '{op}' but found {t!r}" if t else f"expected '{op}' at the end")

    def parse(self):
        node = self.ternary()
        if self.i != len(self.tok):
            self.fail(f"unexpected '{self.peek()[1]}'")
        return node

    def ternary(self):
        c = self.binary(0)
        if self.accept("?"):
            a = self.ternary()
            self.expect(":")
            return ("?", c, a, self.ternary())
        return c

    _LEVELS = [("||",), ("&&",), ("==", "!="), ("<", ">", "<=", ">="), ("+", "-"), ("*", "/")]

    def binary(self, level):
        if level == len(self._LEVELS):
            return self.unary()
        node = self.binary(level + 1)
        while True:
            op = self.accept(*self._LEVELS[level])
            if op is None:
                return node
            node = ("bin", op, node, self.binary(level + 1))

    def unary(self):
        op = self.accept("-", "+", "!")
        if op:
            a = self.unary()
            return a if op == "+" else ("un", op, a)
        return self.primary()

    def primary(self):
        k, t = self.peek()
        if k == "num":
            self.i += 1
            return ("num", float(t))
        if k == "id":
            self.i += 1
            if t == "x" and self.peek() == ("op", "["):
                self.i += 1
                k2, t2 = self.peek()
                if k2 != "num" or not t2.isdigit():
                    self.fail("the index of x must be an integer literal")
                self.i += 1
                self.expect("]")
                return ("x", int(t2))
            if self.peek() == ("op", "("):
                self.i += 1
                if t not in _ARITY:
                    self.fail(f"unknown function '{t}'")
                args = []
                if not self.accept(")"):
                    args.append(self.ternary())
                    while self.accept(","):
                        args.append(self.ternary())
                    self.expect(")")
                lo, hi = _ARITY[t]
                if not lo <= len(args) <= hi:
                    self.fail(f"function '{t}' called with {len(args)} arguments")
                return ("call", t, args)
            return ("id", t)
        if self.accept("("):
            node = self.ternary()
            self.expect(")")
            return node
        self.fail(f"unexpected '{t}'" if t else "unexpected end")


def _names(node, out):
    k = node[0]
    if k == "id":
        out.add(node[1])
    elif k == "un":
        _names(node[2], out)
    elif k == "bin":
        _names(node[2], out)
        _names(node[3], out)
    elif k == "?":
        for c in node[1:]:
            _names(c, out)
    elif k == "call":
        for c in node[2]:
            _names(c, out)
    return out


def parse_cpp(src, params=(), allow_on_boundary=False):
    """Parse a C++ expression string, checking that all identifiers are known."""
    node = _Parser(src).parse()
    known = set(_CONSTS) | set(params) | ({"on_boundary"} if allow_on_boundary else set())
    for n in _names(node, set()):
        if n not in known:
            hint = "only in subdomain conditions" if n == "on_boundary" else \
                f"pass {n}=value to the constructor"
            raise ExpressionError(f"undefined name '{n}' ({hint}) in '{src}'")
    return node


def _div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return math.copysign(math.inf, a) * (math.copysign(1, b)) if a else math.nan


def _safe(f):
    def g(*a):
        try:
            return f(*a)
        except ValueError:
            return math.nan
        except OverflowError:
            return math.inf
    return g


_PYFUNC = {"pow": _safe(math.pow), "sqrt": _safe(math.sqrt), "exp": _safe(math.exp),
           "log": _safe(math.log), "sin": math.sin, "cos": math.cos, "tan": math.tan,
           "asin": _safe(math.asin), "acos": _safe(math.acos), "atan": math.atan,
           "atan2": math.atan2, "sinh": _safe(math.sinh), "cosh": _safe(math.cosh),
           "tanh": math.tanh, "fabs": abs, "abs": abs, "min": min, "max": max, "fmin": min,
           "fmax": max}
_BINPY = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
          "/": _div, "<": lambda a, b: a < b, ">": lambda a, b: a > b,
          "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b,
          "!=": lambda a, b: a != b}


def compile_py(node):
    """AST -> function (x, env) -> number or bool, env maps names to values."""
    k = node[0]
    if k == "num":
        v = node[1]
        return lambda x, env: v
    if k == "x":
        i = node[1]
        return lambda x, env: float(x[i])
    if k == "id":
        n = node[1]
        return lambda x, env: env[n]
    if k == "un":
        a = compile_py(node[2])
        return (lambda x, env: -a(x, env)) if node[1] == "-" else (lambda x, env: not a(x, env))
    if k == "bin":
        op, a, b = node[1], compile_py(node[2]), compile_py(node[3])
        if op == "&&":
            return lambda x, env: bool(a(x, env) and b(x, env))
        if op == "||":
            return lambda x, env: bool(a(x, env) or b(x, env))
        f = _BINPY[op]
        return lambda x, env: f(a(x, env), b(x, env))
    if k == "?":
        c, a, b = (compile_py(n) for n in node[1:])
        return lambda x, env: a(x, env) if c(x, env) else b(x, env)
    name, args = node[1], [compile_py(a) for a in node[2]]
    if name == "near":
        def near_(x, env):
            v = [a(x, env) for a in args]
            return near(v[0], v[1], *v[2:])
        return near_
    f = _PYFUNC[name]
    return lambda x, env: f(*(a(x, env) for a in args))


def _scalar(v):
    return ufl.conditional(v, 1.0, 0.0) if isinstance(v, ufl.classes.Condition) else v


def _cond(v):
    return v if isinstance(v, ufl.classes.Condition) else ufl.ne(_scalar(v), 0.0)


_UFLFUNC = {"sqrt": ufl.sqrt, "exp": ufl.exp, "log": ufl.ln, "sin": ufl.sin, "cos": ufl.cos,
            "tan": ufl.tan, "asin": ufl.asin, "acos": ufl.acos, "atan": ufl.atan,
            "atan2": ufl.atan2, "sinh": ufl.sinh, "cosh": ufl.cosh, "tanh": ufl.tanh,
            "fabs": abs, "abs": abs, "min": ufl.min_value, "max": ufl.max_value,
            "fmin": ufl.min_value, "fmax": ufl.max_value}
_BINUFL = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
           "/": lambda a, b: a / b, "<": ufl.lt, ">": ufl.gt, "<=": ufl.le, ">=": ufl.ge,
           "==": ufl.eq, "!=": ufl.ne}


def to_ufl(node, X, params):
    """AST -> UFL expression over the coordinate X; `params` maps names to Constants."""
    k = node[0]
    if k == "num":
        return node[1]
    if k == "x":
        if node[1] >= X.ufl_shape[0]:
            raise ExpressionError(f"x[{node[1]}] on a mesh of dimension {X.ufl_shape[0]}")
        return X[node[1]]
    if k == "id":
        return _CONSTS[node[1]] if node[1] in _CONSTS else params[node[1]]
    if k == "un":
        a = to_ufl(node[2], X, params)
        return -_scalar(a) if node[1] == "-" else ufl.Not(_cond(a))
    if k == "bin":
        op = node[1]
        a, b = to_ufl(node[2], X, params), to_ufl(node[3], X, params)
        if op == "&&":
            return ufl.And(_cond(a), _cond(b))
        if op == "||":
            return ufl.Or(_cond(a), _cond(b))
        return _BINUFL[op](_scalar(a), _scalar(b))
    if k == "?":
        c, a, b = (to_ufl(n, X, params) for n in node[1:])
        return ufl.conditional(_cond(c), _scalar(a), _scalar(b))
    name, args = node[1], [_scalar(to_ufl(a, X, params)) for a in node[2]]
    if name == "pow":
        return ufl.as_ufl(args[0]) ** args[1]
    if name == "near":
        eps = args[2] if len(args) > 2 else DOLFIN_EPS
        return ufl.And(ufl.ge(args[0], args[1] - eps), ufl.le(args[0], args[1] + eps))
    return _UFLFUNC[name](*args)


def _is_code(code):
    return isinstance(code, str)


def _lists(code):
    return [_lists(c) for c in code] if isinstance(code, (tuple, list)) else code


def _map_tree(f, code):
    return [_map_tree(f, c) for c in code] if isinstance(code, list) else f(code)


def _as_code(c):
    return c if isinstance(c, str) else repr(float(c))


class _FloatingSpace(ufl.FunctionSpace):
    """Space whose domain is the mesh created last, so an Expression can outlive its mesh."""

    def ufl_domain(self):
        return default_mesh().ufl_domain()

    def ufl_domains(self):
        return (self.ufl_domain(),)


class Expression(ExternalFunction):
    """``Expression("1 + x[0]*x[0]", degree=2)``, vector valued with a tuple of strings.

    Extra keyword arguments are parameters (``Expression("exp(-a*x[0])", a=5, degree=1)``),
    changed by attribute: ``expr.a = 2``.  Without `mesh=` it follows the mesh created last.
    """

    def __new__(cls, cppcode=None, *args, **kwargs):
        if cppcode is None:
            raise NotImplementedError("Expression subclasses with eval are not supported, "
                                      "give the C++ expression string")
        return super().__new__(cls, cppcode, *args, **kwargs)

    def __init__(self, cppcode=None, *args, degree=None, element=None, domain=None, mesh=None,
                 cell=None, mpi_comm=None, **params):
        if args:
            degree = args[0] if degree is None else degree
        if degree is None and element is not None:
            degree = element.embedded_superdegree
        if degree is None:
            raise ValueError("Expression needs degree= (or element=)")
        m = mesh if mesh is not None else domain
        floating = m is None
        if m is None:
            m = default_mesh()
        elif not isinstance(m, _CoreMesh):
            m = ufl.domain.extract_unique_domain(m).ufl_cargo()
        for n, v in params.items():
            if not isinstance(v, numbers.Number):
                raise TypeError(f"parameter '{n}' must be a number")
        code = _map_tree(_as_code, _lists(cppcode))
        asts = _map_tree(lambda c: parse_cpp(c, params), code)
        consts = {n: Constant(m, float(v)) for n, v in params.items()}
        X = ufl.SpatialCoordinate(m.ufl_domain())
        built = _map_tree(lambda a: ufl.as_ufl(_scalar(to_ufl(a, X, consts))), asts)
        expr = ufl.as_tensor(built) if isinstance(built, list) else built
        super().__init__(as_cf(expr, m), m, int(degree))
        if floating:
            self._ufl_function_space = _FloatingSpace(m.ufl_domain(), self.ufl_element())
        self.__dict__.update(_params=consts, _code=cppcode, _fun=_map_tree(compile_py, asts),
                             _gdim=m.dim, _mesh=m)

    def __getattribute__(self, name):
        p = object.__getattribute__(self, "__dict__").get("_params")
        if p and name in p:
            return p[name].value
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        p = self.__dict__.get("_params")
        if p and name in p:
            p[name].assign(float(value))
        else:
            object.__setattr__(self, name, value)

    def _env(self):
        return {n: c.value for n, c in self._params.items()} | _CONSTS

    def _eval(self, x, env):
        x = np.asarray(x, dtype=float).ravel()
        if isinstance(self._fun, list):
            return np.array(_map_tree(lambda f: float(f(x, env)), self._fun))
        return self._fun(x, env)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            return super().__call__(*args, **kwargs)
        x = args[0] if len(args) == 1 else args
        v = self._eval(x, self._env())
        return float(v) if np.ndim(v) == 0 else v

    def eval(self, values, x):
        values[:] = np.atleast_1d(self(x))

    def compute_vertex_values(self, mesh=None):
        m = mesh or self._mesh
        pts = np.array([list(v.point)[:m.dim] for v in m.ngs.vertices])
        env = self._env()
        vals = np.array([np.atleast_1d(self._eval(p, env)) for p in pts], dtype=float)
        return vals.T.ravel()

    def geometric_dimension(self):
        return self._gdim

    def value_rank(self):
        return len(self.ufl_shape)

    def value_dimension(self, i):
        return self.ufl_shape[i]


# ---------------------------------------------------------------- meshes

class Point:
    def __init__(self, x, y=0.0, z=0.0):
        if isinstance(x, (tuple, list, np.ndarray, Point)):
            x = list(x)
            c = x + [0.0] * (3 - len(x))
            x, y, z = c[:3]
        self._c = np.array([x, y, z], dtype=float)

    def x(self):
        return float(self._c[0])

    def y(self):
        return float(self._c[1])

    def z(self):
        return float(self._c[2])

    def array(self):
        return self._c.copy()

    def __getitem__(self, i):
        return float(self._c[i])

    def __len__(self):
        return 3

    def __iter__(self):
        return iter(self._c.tolist())

    def distance(self, other):
        return float(np.linalg.norm(self._c - np.asarray(list(other))))

    def norm(self):
        return float(np.linalg.norm(self._c))

    def dot(self, other):
        return float(self._c @ np.asarray(list(other)))

    def __repr__(self):
        return f"Point({self._c[0]}, {self._c[1]}, {self._c[2]})"


class _Dim:
    def __init__(self, d):
        self._d = d

    def dim(self):
        return self._d


class Mesh(_CoreMesh):
    """Mesh with the dolfin accessors ``coordinates()``, ``cells()``, ``hmax()``, ..."""

    def coordinates(self):
        if "_xyz" not in self._quantities:
            self._quantities["_xyz"] = np.array(
                [list(v.point)[:self.dim] for v in self.ngs.vertices], dtype=float)
        return self._quantities["_xyz"].copy()

    def cells(self):
        if "_cells" not in self._quantities:
            self._quantities["_cells"] = np.array(
                [[v.nr for v in el.vertices] for el in self.ngs.Elements(ngs.VOL)], dtype=int)
        return self._quantities["_cells"].copy()

    def _diameters(self):
        X, C = self.coordinates(), self.cells()
        d = np.zeros(len(C))
        for a in range(C.shape[1]):
            for b in range(a + 1, C.shape[1]):
                d = np.maximum(d, np.linalg.norm(X[C[:, a]] - X[C[:, b]], axis=1))
        return d

    def hmax(self):
        return float(self._diameters().max())

    def hmin(self):
        return float(self._diameters().min())

    def geometry(self):
        return _Dim(self.dim)

    def topology(self):
        return _Dim(self.dim)

    def init(self, *args):
        pass

    def mpi_comm(self):
        return None


def _diag(diagonal):
    if diagonal in ("right", None):
        return True
    if diagonal == "left":
        return False
    raise NotImplementedError(f"diagonal='{diagonal}'; only 'left' and 'right'")


def UnitIntervalMesh(nx):
    return _meshmod.interval(nx, 0.0, 1.0, cls=Mesh)


def IntervalMesh(nx, a, b):
    return _meshmod.interval(nx, a, b, cls=Mesh)


def RectangleMesh(p0, p1, nx, ny, diagonal="right"):
    p0, p1 = Point(p0), Point(p1)
    return _meshmod.rectangle(nx, ny, (p0[0], p0[1]), (p1[0], p1[1]), flip=_diag(diagonal), cls=Mesh)


def UnitSquareMesh(nx, ny, diagonal="right"):
    return RectangleMesh(Point(0, 0), Point(1, 1), nx, ny, diagonal)


def BoxMesh(p0, p1, nx, ny, nz):
    p0, p1 = Point(p0), Point(p1)
    return _meshmod.box(nx, ny, nz, tuple(p0), tuple(p1), cls=Mesh)


def UnitCubeMesh(nx, ny, nz):
    return BoxMesh(Point(0, 0, 0), Point(1, 1, 1), nx, ny, nz)


# ---------------------------------------------------------------- sub domains and boundary conditions

class SubDomain:
    def __init__(self, map_tol=1e-10):
        pass

    def inside(self, x, on_boundary):
        raise NotImplementedError("override inside(self, x, on_boundary)")

    def mark(self, *args, **kwargs):
        raise NotImplementedError("SubDomain.mark needs MeshFunction, which is not available "
                                  "in ngsufl.fenics yet; use DirichletBC(V, g, subdomain)")

    def map(self, x, y):
        raise NotImplementedError("periodic maps are not available in ngsufl.fenics yet")


class AutoSubDomain(SubDomain):
    def __init__(self, func):
        super().__init__()
        self._func = func

    def inside(self, x, on_boundary):
        return self._func(x, on_boundary)


class CompiledSubDomain(SubDomain):
    """``CompiledSubDomain("near(x[0], side)", side=0.0)``; parameters change by attribute."""

    def __init__(self, cppcode, **params):
        super().__init__()
        self.__dict__["_params"] = dict(params)
        self.__dict__["_fun"] = compile_py(parse_cpp(cppcode, params, allow_on_boundary=True))

    def set_property(self, name, value):
        self._params[name] = value

    def __setattr__(self, name, value):
        if name in self._params:
            self._params[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        p = self.__dict__.get("_params", {})
        if name in p:
            return p[name]
        raise AttributeError(name)

    def inside(self, x, on_boundary):
        return bool(self._fun(x, {**_CONSTS, **self._params, "on_boundary": bool(on_boundary)}))


def _where(V, sub_domain):
    if isinstance(sub_domain, SubDomain):
        return lambda x, on_boundary: bool(sub_domain.inside(x, on_boundary))
    if isinstance(sub_domain, str):
        if sub_domain == "on_boundary":
            return sub_domain
        try:
            fun = compile_py(parse_cpp(sub_domain, allow_on_boundary=True))
        except ExpressionError:
            if re.fullmatch(r"[\w|.*]+", sub_domain) and V.mesh.ngs.GetBoundaries():
                return sub_domain
            raise
        return lambda x, on_boundary: bool(fun(x, {**_CONSTS, "on_boundary": on_boundary}))
    return sub_domain


class DirichletBC(_CoreBC):
    """``DirichletBC(V, g, sub_domain, method="topological")``, see the module docstring."""

    def __init__(self, V, g, sub_domain="on_boundary", method="topological", check_midpoint=True):
        if method not in ("topological", "geometric", "pointwise"):
            raise ValueError(f"unknown method '{method}'")
        super().__init__(V, g, _where(V, sub_domain))
        self.method = method

    def set_value(self, g):
        self.g = g

    def homogenize(self):
        self.g = 0.0

    def get_boundary_values(self):
        f = Function(self.space.root)
        self.apply(f)
        vals = f.vector().get_local()
        return {int(i): float(vals[i]) for i, b in enumerate(self.dofs()) if b}


# ---------------------------------------------------------------- arguments and functions

class _Argument(ufl.Argument):
    """UFL argument that also answers ``geometric_dimension()`` like the old UFL did."""

    def geometric_dimension(self):
        V = self.ufl_function_space()
        return V.mesh.dim if hasattr(V, "mesh") else V.ufl_domain().geometric_dimension()

    def __eq__(self, other):
        return (isinstance(other, ufl.Argument) and self.number() == other.number()
                and self.part() == other.part()
                and self.ufl_function_space() == other.ufl_function_space())

    __hash__ = ufl.Argument.__hash__


def TestFunction(V):
    return _Argument(V, 0)


def TrialFunction(V):
    return _Argument(V, 1)


def TestFunctions(V):
    return ufl.split(TestFunction(V))


def TrialFunctions(V):
    return ufl.split(TrialFunction(V))


class Function(_CoreFunction):
    """Function with ``split(deepcopy=...)`` like dolfin."""

    def split(self, deepcopy=False):
        out = []
        for s in self.subfunctions:
            f = Function(s.function_space(), name=s.name, gf=s.gf)
            out.append(f.copy() if deepcopy else f)
        return tuple(out)

    def copy(self, deepcopy=True):
        f = Function(self.function_space(), name=self.name)
        f.gf.vec.data = self.gf.vec
        return f


def FunctionSpace(mesh, *args, **kwargs):
    if kwargs.pop("constrained_domain", None) is not None:
        raise NotImplementedError("constrained_domain (periodic spaces) is not available "
                                  "in ngsufl.fenics yet")
    return _CoreFunctionSpace(mesh, *args, **kwargs)


def VectorFunctionSpace(mesh, *args, **kwargs):
    if kwargs.pop("constrained_domain", None) is not None:
        raise NotImplementedError("constrained_domain (periodic spaces) is not available "
                                  "in ngsufl.fenics yet")
    return _core.VectorFunctionSpace(mesh, *args, **kwargs)


def TensorFunctionSpace(mesh, *args, **kwargs):
    return _core.TensorFunctionSpace(mesh, *args, **kwargs)


def interpolate(expr, V):
    return Function(V).interpolate(expr)


def project(v, V=None, bcs=None, mesh=None, function=None, solver_type="lu",
            preconditioner_type="default", form_compiler_parameters=None):
    if V is None:
        if not isinstance(v, _CoreFunction):
            raise ValueError("project needs the target space V")
        V = v.function_space()
    sp = _linear_options(solver_type, preconditioner_type)
    w = _solving.project(v, V, bcs, solver_parameters=sp)
    out = function if function is not None else Function(V)
    out.gf.vec.data = w.gf.vec
    return out


def norm(v, norm_type="L2", mesh=None):
    if isinstance(v, Vector):
        return v.norm(norm_type)
    if isinstance(v, _CoreFunction) and norm_type in ("l2", "l1", "linf"):
        return v.vector().norm(norm_type)
    return _solving.norm(v, norm_type)


def errornorm(u, uh, norm_type="L2", degree_rise=3, mesh=None):
    """Norm of u - uh, both interpolated into a space `degree_rise` degrees higher."""
    el = uh.ufl_element() if isinstance(uh, _CoreFunction) else None
    if el is not None and getattr(el, "family", None) in ("Lagrange", "DG"):
        V = uh.function_space()
        Vh = _core.FunctionSpace(V.mesh, el.family, el.degree + degree_rise, shape=el.shape or None)
        return _solving.norm(Function(Vh).interpolate(u) - Function(Vh).interpolate(uh), norm_type)
    return _solving.errornorm(u, uh, norm_type)


def assemble(form, tensor=None, bcs=None, **kwargs):
    out = _solving.assemble(form, bcs=bcs, **{k: v for k, v in kwargs.items() if k not in
                                              ("form_compiler_parameters", "keep_diagonal",
                                               "add_values", "finalize_tensor", "backend",
                                               "mesh", "cell_domains", "exterior_facet_domains",
                                               "interior_facet_domains")})
    if tensor is None:
        return out
    if isinstance(out, Vector):
        tensor.vec.data = out.vec
    elif isinstance(out, Matrix):
        tensor.__dict__.update(out.__dict__)
    else:
        raise TypeError("tensor= needs a vector or matrix form")
    return tensor


# ---------------------------------------------------------------- solvers

_LU = {"lu", "default", "mumps", "umfpack", "superlu", "superlu_dist", "petsc", "cholesky",
       "pastix", "pardiso"}
_KRYLOV_NAMES = {"cg": "cg", "gmres": "gmres", "minres": "minres", "bicgstab": "gmres",
                 "tfqmr": "gmres", "bicg": "gmres"}


def _linear_options(solver="default", preconditioner="default"):
    s = str(solver or "default").lower()
    if s in _LU:
        return {"ksp_type": "preonly", "pc_type": "lu"}
    if s not in _KRYLOV_NAMES:
        raise NotImplementedError(f"linear_solver '{solver}'; use lu, cg, gmres or minres")
    pc = str(preconditioner or "default").lower()
    return {"ksp_type": _KRYLOV_NAMES[s],
            "pc_type": pc if pc in ("none", "lu", "jacobi") else "jacobi"}


_NEWTON = {"relative_tolerance": "snes_rtol", "absolute_tolerance": "snes_atol",
           "maximum_iterations": "snes_max_it", "relaxation_parameter": "snes_damping",
           }
_KRYLOV_KEYS = {"relative_tolerance": "ksp_rtol", "maximum_iterations": "ksp_max_it",
                "monitor_convergence": "ksp_monitor"}


def _convert(sp):
    """Legacy solver_parameters -> options of the core solvers."""
    sp = sp or {}
    out = {k: v for k, v in sp.items() if k.startswith(("ksp_", "pc_", "snes_"))}
    nl = sp.get("newton_solver") or sp.get("snes_solver") or {}
    solver = nl.get("linear_solver", sp.get("linear_solver", "default"))
    pc = nl.get("preconditioner", sp.get("preconditioner", "default"))
    out = {**_linear_options(solver, pc), **out}
    for src in (sp.get("krylov_solver") or {}, nl.get("krylov_solver") or {}):
        for k, v in src.items():
            if k in _KRYLOV_KEYS:
                out[_KRYLOV_KEYS[k]] = v
    for k, v in nl.items():
        if k in _NEWTON:
            out[_NEWTON[k]] = v
    return out


def solve(*args, **kwargs):
    """``solve(a == L, u, bcs)``, ``solve(F == 0, u, bcs, J=J)``, ``solve(A, x, b, "cg", "ilu")``."""
    if args and isinstance(args[0], Matrix):
        A, x, b, *rest = args
        solver = rest[0] if rest else kwargs.get("linear_solver", "default")
        pc = rest[1] if len(rest) > 1 else kwargs.get("preconditioner", "default")
        _solving.LinearSolver(A, _linear_options(solver, pc)).solve(x, b)
        return
    names = ["equation", "u", "bcs", "J", "tol", "M", "form_compiler_parameters", "solver_parameters"]
    opts = dict(zip(names, args))
    opts.update(kwargs)
    if opts.get("M") is not None or opts.get("tol") is not None:
        raise NotImplementedError("adaptive solves (tol, M) are not available")
    _solving.solve(opts["equation"], opts["u"], opts.get("bcs"), J=opts.get("J"),
                   solver_parameters=_convert(opts.get("solver_parameters")))


LinearVariationalProblem = _solving.LinearVariationalProblem
NonlinearVariationalProblem = _solving.NonlinearVariationalProblem

_LIN_DEFAULTS = {"linear_solver": "default", "preconditioner": "default", "symmetric": False,
                 "krylov_solver": {"relative_tolerance": 1e-10, "maximum_iterations": 1000,
                                   "monitor_convergence": False},
                 "lu_solver": {"report": False}}
_NL_DEFAULTS = {"nonlinear_solver": "newton",
                "newton_solver": {"relative_tolerance": 1e-9, "absolute_tolerance": 1e-10,
                                  "maximum_iterations": 50, "relaxation_parameter": 1.0,
                                  "error_on_nonconvergence": True, "convergence_criterion": "residual",
                                  "report": False, "linear_solver": "default",
                                  "preconditioner": "default",
                                  "krylov_solver": {}, "lu_solver": {}}}


class LinearVariationalSolver:
    def __init__(self, problem):
        self.problem = problem
        self.parameters = _nested(_LIN_DEFAULTS)
        self._core = None

    def solve(self):
        opts = _convert(self.parameters)
        if self._core is None:
            self._core = _solving.LinearVariationalSolver(self.problem, opts)
        self._core.params = opts
        self._core.solve()
        return 1, True


class NonlinearVariationalSolver:
    """Newton solver; ``solve()`` returns ``(iterations, converged)``."""

    def __init__(self, problem):
        self.problem = problem
        self.parameters = _nested(_NL_DEFAULTS)
        self._core = None
        self.iterations = 0

    def solve(self):
        nl = self.parameters["newton_solver"]
        opts = _convert(self.parameters)
        if self._core is None:
            self._core = _solving.NonlinearVariationalSolver(self.problem, opts)
        self._core.params = opts
        try:
            self._core.solve()
        except ConvergenceError:
            if nl.get("error_on_nonconvergence", True):
                raise
            self.iterations = int(nl["maximum_iterations"])
            return self.iterations, False
        self.iterations = self._core.iterations
        return self.iterations, True


def assemble_system(a, L, bcs=None, **kwargs):
    A, b = assemble(a), assemble(L)
    for bc in ([] if bcs is None else [bcs] if isinstance(bcs, _CoreBC) else bcs):
        bc.apply(A, b)
    return A, b


class LUSolver:
    def __init__(self, *args):
        self.A = args[0] if args and isinstance(args[0], Matrix) else None
        self.method = next((a for a in args if isinstance(a, str)), "default")
        self.parameters = _nested({"report": False, "symmetric": False})

    def set_operator(self, A):
        self.A = A

    def solve(self, *args):
        if len(args) == 3:
            self.A, x, b = args
        else:
            x, b = args
        _solving.LinearSolver(self.A, _linear_options("lu")).solve(x, b)
        return 1


class KrylovSolver:
    def __init__(self, *args):
        strs = [a for a in args if isinstance(a, str)]
        self.A = next((a for a in args if isinstance(a, Matrix)), None)
        self.method = strs[0] if strs else "default"
        self.pc = strs[1] if len(strs) > 1 else "default"
        self.parameters = _nested({"relative_tolerance": 1e-10, "maximum_iterations": 1000,
                                   "monitor_convergence": False})

    def set_operator(self, A):
        self.A = A

    def solve(self, *args):
        if len(args) == 3:
            self.A, x, b = args
        else:
            x, b = args
        opts = _linear_options(self.method, self.pc)
        opts["ksp_rtol"] = self.parameters["relative_tolerance"]
        opts["ksp_max_it"] = self.parameters["maximum_iterations"]
        _solving.LinearSolver(self.A, opts).solve(x, b)
        return 1


# ---------------------------------------------------------------- output

_PLOT_IGNORED = ("interactive", "scalarbar", "backend", "window_title", "wireframe",
                 "range_min", "range_max", "prefix", "rescale", "axes_labels")


def plot(obj, title=None, **kwargs):
    """Draw on the current axes: mesh -> wireframe, scalar -> colours, vector -> arrows.

    ``mode="displacement"`` draws the mesh moved by a 2D vector field, ``mode="color"`` the
    magnitude of a vector field.  No colorbar is added, use ``plt.colorbar(plot(u))``.
    """
    import matplotlib.pyplot as plt
    ax = kwargs.pop("ax", None) or kwargs.pop("axes", None)
    for k in _PLOT_IGNORED:
        kwargs.pop(k, None)
    mode = kwargs.pop("mode", None)
    if isinstance(obj, _CoreMesh):
        if obj.dim != 2:
            raise NotImplementedError("mesh plot needs a 2D mesh")
        ax = ax or plt.gca()
        kw = {"color": kwargs.pop("color", "k"), "linewidth": kwargs.pop("linewidth", 0.5)}
        art = _pyplot.triplot(obj, axes=ax, interior_kw=kw,
                              boundary_kw={"colors": [kw["color"]], "linewidth": kw["linewidth"]})[0]
        ax.set_aspect("equal")
    else:
        if isinstance(obj, _CoreFunction):
            mesh = obj.function_space().mesh
        else:
            mesh = ufl.domain.extract_unique_domain(obj).ufl_cargo()
        vector = len(obj.ufl_shape) > 0 and int(np.prod(obj.ufl_shape)) > 1
        if mesh.dim == 3:
            fig = plt.gcf()
            ax3 = ax or (fig.axes[-1] if fig.axes and hasattr(fig.axes[-1], "get_zlim") else None)
            kwargs.setdefault("colorbar", False)
            ax = art = _output.plot(obj, ax=ax3, **kwargs)
        else:
            ax = ax or plt.gca()
            if mesh.dim == 1:
                art = _pyplot.plot(obj, axes=ax, **kwargs)[0]
            elif vector and mode == "displacement":
                xy, vals, tris = _output._sample(mesh, _field_cf(obj, mesh), 1)
                from matplotlib.tri import Triangulation
                art = ax.triplot(Triangulation(*(xy + vals[:, :2]).T, tris),
                                 color=kwargs.pop("color", "k"), lw=kwargs.pop("linewidth", 0.5))[0]
                ax.set_aspect("equal")
            elif vector and mode != "color":
                art = _pyplot.quiver(obj, axes=ax, **kwargs)
                ax.set_aspect("equal")
            elif "levels" in kwargs:
                art = _pyplot.tricontourf(obj, axes=ax, **kwargs)
                ax.set_aspect("equal")
            else:
                art = _pyplot.tripcolor(obj, axes=ax, **kwargs)
                ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    return art


def _field_cf(obj, mesh):
    return obj.cf if isinstance(obj, _CoreFunction) else to_cf(obj, mesh)


class File:
    """``File("out/u.pvd") << u`` or ``<< (u, t)``; writes VTK files."""

    def __init__(self, filename, *args, **kwargs):
        if not filename.endswith((".pvd", ".vtu", ".vtk")):
            raise NotImplementedError(f"File('{filename}'): only VTK output (.pvd) is available")
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        self._vtk = VTKFile(filename)

    def __lshift__(self, obj):
        t = None
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], numbers.Number):
            obj, t = obj
        if isinstance(obj, _CoreMesh):
            raise NotImplementedError("writing a mesh is not available in ngsufl.fenics yet")
        if not isinstance(obj, _CoreFunction):
            raise TypeError("File << expects a Function (or a pair (Function, time))")
        self._vtk.write(obj, time=t)
        return self

    def write(self, obj, t=None):
        return self << (obj, t) if t is not None else self << obj


__all__ = sorted(set(_core_all) | {
    "DOLFIN_EPS", "DOLFIN_PI", "near", "between", "parameters", "LogLevel", "set_log_level",
    "set_log_active", "info", "interactive", "MPI", "mpi_comm_world", "Expression", "Point",
    "Mesh", "SubDomain", "AutoSubDomain", "CompiledSubDomain", "TestFunction", "TrialFunction",
    "TestFunctions", "TrialFunctions", "Function", "FunctionSpace", "VectorFunctionSpace",
    "TensorFunctionSpace", "interpolate", "project", "norm", "errornorm", "assemble", "solve",
    "assemble_system", "LinearVariationalSolver", "NonlinearVariationalSolver", "LUSolver",
    "KrylovSolver", "plot", "File", "DirichletBC", "UnitIntervalMesh", "IntervalMesh",
    "UnitSquareMesh", "RectangleMesh", "UnitCubeMesh", "BoxMesh", "LinearVariationalProblem",
    "NonlinearVariationalProblem"} | set(_STUBS))

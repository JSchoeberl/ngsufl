"""Function spaces, functions and constants."""

import numbers

import ngsolve as ngs
import numpy as np
import ufl
from ufl.constantvalue import ConstantValue
from ufl.core.ufl_type import ufl_type
from ufl.finiteelement import AbstractFiniteElement
from ufl.utils.counted import Counted

from .element import FiniteElement, MixedElement, make_fespace
from .mesh import Mesh, default_mesh
from .translate import Field, Part, translate


class FunctionSpace(ufl.FunctionSpace):
    """``FunctionSpace(mesh, "Lagrange", 2)`` or ``FunctionSpace(mesh, ("Lagrange", 2, (dim,)))``.

    Extra keyword arguments are passed to the NGSolve space.
    """

    def __init__(self, mesh, family, degree=None, shape=None, **flags):
        if isinstance(family, AbstractFiniteElement):
            element = family
        else:
            if isinstance(family, (tuple, list)):
                family, degree, *rest = family
                shape = rest[0] if rest else shape
            element = FiniteElement(family, mesh.ufl_cell(), degree, shape=shape)
        super().__init__(mesh.ufl_domain(), element)
        self.mesh = mesh
        self.flags = flags
        self.fes = make_fespace(mesh.ngs, element, **flags)
        self._dg = None

    def ngs_space(self, dg=False):
        """The NGSolve space; with dg=True a twin that couples across facets."""
        if not dg or self.flags.get("dgjumps"):
            return self.fes
        if self._dg is None:
            self._dg = make_fespace(self.mesh.ngs, self.ufl_element(), dgjumps=True, **self.flags)
        return self._dg

    def __call__(self):
        return self

    def dim(self):
        return self.fes.ndof

    def num_sub_spaces(self):
        return len(self.ufl_element().sub_elements)

    @property
    def subspaces(self):
        if isinstance(self.ufl_element(), MixedElement):
            return tuple(self.sub(i) for i in range(self.num_sub_spaces()))
        return (self,)

    def __len__(self):
        return len(self.ufl_element().sub_elements) if isinstance(self.ufl_element(), MixedElement) else 1

    def sub(self, i):
        if i == 0 and not self.ufl_element().sub_elements:
            return self
        return SubSpace(self, (i,))

    __getitem__ = sub

    def dof_range(self):
        return range(self.fes.ndof)

    @property
    def root(self):
        return self

    def collapse(self):
        return self

    def __mul__(self, other):
        return MixedFunctionSpace([self, other])


def _flat_elements(element):
    return list(element.sub_elements) if isinstance(element, MixedElement) else [element]


def MixedFunctionSpace(spaces):
    """Product of spaces, also available as ``V * Q``."""
    elements, flags = [], {}
    for V in spaces:
        elements += _flat_elements(V.ufl_element())
        flags.update(V.flags)
    return FunctionSpace(spaces[0].mesh, MixedElement(elements), **flags)


def VectorFunctionSpace(mesh, family, degree=None, dim=None, **flags):
    if isinstance(family, (tuple, list)):
        family, degree = family[:2]
    return FunctionSpace(mesh, family, degree, shape=(dim or mesh.dim,), **flags)


def TensorFunctionSpace(mesh, family, degree=None, shape=None, **flags):
    if isinstance(family, (tuple, list)):
        family, degree = family[:2]
    return FunctionSpace(mesh, family, degree, shape=shape or (mesh.dim, mesh.dim), **flags)


class SubSpace:
    """View of a component of a space, used for boundary conditions."""

    def __init__(self, root, path):
        self.root, self.path, self.mesh = root, tuple(path), root.mesh

    def _walk(self):
        fes, element, start = self.root.fes, self.root.ufl_element(), 0
        for i in self.path:
            start += fes.Range(i).start
            fes, element = fes.components[i], element.sub_elements[i]
        return fes, element, start

    @property
    def fes(self):
        return self._walk()[0]

    def ufl_element(self):
        return self._walk()[1]

    def dof_range(self):
        fes, _, start = self._walk()
        return range(start, start + fes.ndof)

    def sub(self, i):
        return SubSpace(self.root, self.path + (i,))

    def collapse(self):
        return FunctionSpace(self.mesh, self.ufl_element(), **self.root.flags)


class Argument(ufl.Argument):
    """UFL argument with ``function_space()`` as in Firedrake."""

    def function_space(self):
        return self.ufl_function_space()

    def geometric_dimension(self):
        return self.ufl_function_space().mesh.dim

    def __eq__(self, other):
        return (isinstance(other, ufl.Argument) and self.number() == other.number()
                and self.part() == other.part()
                and self.ufl_function_space() == other.ufl_function_space())

    __hash__ = ufl.Argument.__hash__


def TestFunction(V, part=None):
    return Argument(V, 0, part)


def TrialFunction(V, part=None):
    return Argument(V, 1, part)


def TestFunctions(V):
    return ufl.split(TestFunction(V))


def TrialFunctions(V):
    return ufl.split(TrialFunction(V))


class _Sym:
    """Lets numpy style callables (``lambda x: np.sin(x[0])``) build UFL expressions."""

    __array_priority__ = 1000

    def __init__(self, e):
        self.e = e

    def __getitem__(self, i):
        return _Sym(self.e[i])

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if method != "__call__" or ufunc.__name__ not in _UFUNCS:
            return NotImplemented
        return _Sym(_UFUNCS[ufunc.__name__](*inputs))

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "where" and len(args) == 3:
            return _Sym(ufl.conditional(_cond(args[0]), _unwrap(args[1]), _unwrap(args[2])))
        if func.__name__ in ("zeros_like", "ones_like", "full_like"):
            value = {"zeros_like": 0.0, "ones_like": 1.0}.get(func.__name__, args[1] if len(args) > 1 else kwargs.get("fill_value"))
            return _Sym(ufl.as_ufl(float(value)))
        if func.__name__ in ("stack", "vstack", "array", "asarray"):
            return _Sym(_unwrap(list(args[0])))
        return NotImplemented


def _is_cond(e):
    return isinstance(e, ufl.classes.Condition)


def _unwrap(a):
    if isinstance(a, _Sym):
        return ufl.conditional(a.e, 1.0, 0.0) if _is_cond(a.e) else a.e
    if isinstance(a, np.generic):
        return a.item()
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(a, (tuple, list)):
        return ufl.as_tensor([_unwrap(b) for b in a])
    return a


def _cond(a):
    if isinstance(a, _Sym) and _is_cond(a.e):
        return a.e
    return ufl.ne(_unwrap(a), 0)


_UFUNCS = {
    "sin": ufl.sin, "cos": ufl.cos, "tan": ufl.tan, "exp": ufl.exp, "log": ufl.ln, "sqrt": ufl.sqrt,
    "absolute": abs, "arctan": ufl.atan, "arcsin": ufl.asin, "arccos": ufl.acos, "sinh": ufl.sinh,
    "cosh": ufl.cosh, "tanh": ufl.tanh, "arctan2": ufl.atan2, "maximum": ufl.max_value,
    "minimum": ufl.min_value, "add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b, "true_divide": lambda a, b: a / b, "power": lambda a, b: a ** b,
    "negative": lambda a: -a, "square": lambda a: a * a,
    "less": ufl.lt, "less_equal": ufl.le, "greater": ufl.gt, "greater_equal": ufl.ge,
}
_UFUNCS = {k: (lambda f: lambda *args: f(*map(_unwrap, args)))(v) for k, v in _UFUNCS.items()}
_UFUNCS.update({
    "logical_and": lambda a, b: ufl.And(_cond(a), _cond(b)),
    "logical_or": lambda a, b: ufl.Or(_cond(a), _cond(b)),
    "logical_not": lambda a: ufl.Not(_cond(a)),
    "bitwise_and": lambda a, b: ufl.And(_cond(a), _cond(b)),
    "bitwise_or": lambda a, b: ufl.Or(_cond(a), _cond(b)),
})


def _binop(expr):
    def op(self, other):
        return _Sym(expr(_unwrap(self), _unwrap(other)))
    return op


for _n, _f in {"add": lambda a, b: a + b, "radd": lambda a, b: b + a, "sub": lambda a, b: a - b,
               "rsub": lambda a, b: b - a, "mul": lambda a, b: a * b, "rmul": lambda a, b: b * a,
               "truediv": lambda a, b: a / b, "rtruediv": lambda a, b: b / a,
               "pow": lambda a, b: a ** b, "rpow": lambda a, b: b ** a,
               "lt": ufl.lt, "le": ufl.le, "gt": ufl.gt, "ge": ufl.ge}.items():
    setattr(_Sym, f"__{_n}__", _binop(_f))
_Sym.__neg__ = lambda self: _Sym(-_unwrap(self))
_Sym.__abs__ = lambda self: _Sym(abs(_unwrap(self)))
_Sym.__and__ = lambda self, other: _Sym(ufl.And(_cond(self), _cond(other)))
_Sym.__or__ = lambda self, other: _Sym(ufl.Or(_cond(self), _cond(other)))
_Sym.__invert__ = lambda self: _Sym(ufl.Not(_cond(self)))


def set_values(gf, cf, element, definedon=None):
    """Canonical interpolation for H1, H(curl) and H(div), local projection otherwise."""
    kw = {} if definedon is None else {"definedon": definedon}
    if getattr(element, "family", None) in ("Lagrange", "N1curl", "N2curl", "RT", "BDM"):
        try:
            gf.Interpolate(cf, **kw)
            return
        except Exception:
            pass
    gf.Set(cf, **kw)


def as_cf(g, mesh, shape=()):
    """Number, tuple, callable, UFL expression or NGSolve CF -> CoefficientFunction."""
    if isinstance(g, ngs.CoefficientFunction):
        return g
    if callable(g) and not isinstance(g, ufl.core.expr.Expr):
        g = _unwrap(g(_Sym(ufl.SpatialCoordinate(mesh.ufl_domain()))))
    if isinstance(g, numbers.Number):
        if not shape:
            return ngs.CF(float(g))
        g = np.full(shape, float(g))
    if isinstance(g, (tuple, list, np.ndarray)):
        arr = np.asarray(g)
        if arr.dtype != object:
            cf = ngs.CF(tuple(float(v) for v in arr.flat))
            return cf if arr.ndim == 1 else cf.Reshape(arr.shape)
        g = ufl.as_tensor(g)
    cf = translate(g, mesh)
    return ngs.CF(cf) if isinstance(cf, numbers.Number) else cf


def to_cf(expr, mesh=None):
    """Translate a UFL expression (without trial/test functions) to an NGSolve CF."""
    if mesh is None:
        try:
            mesh = ufl.domain.extract_unique_domain(expr).ufl_cargo()
        except Exception:
            mesh = default_mesh()
    return as_cf(expr, mesh)


class _Name(str):
    """A str that may also be called, ``u.name`` and ``u.name()``."""

    def __call__(self):
        return str(self)


def _columns(element):
    """Number of components that have their own block of dofs."""
    if isinstance(element, FiniteElement) and element.family in ("Lagrange", "DG", "CR", "Real"):
        return int(np.prod(element.shape, dtype=int)), element.shape
    return 1, ()


class Dat:
    """``u.dat.data``: the coefficient array, one row per dof for vector spaces.

    The coefficients belong to NGSolve's hierarchical basis.  They are point values
    only for lowest order spaces.
    """

    def __init__(self, vec, element=None, parts=None):
        self._vec, self._element, self._parts = vec, element, parts

    @property
    def data(self):
        if self._parts is not None:
            return tuple(p.dat.data for p in self._parts)
        arr = self._vec.FV().NumPy()
        n, shape = _columns(self._element)
        if n == 1:
            return arr
        arr = arr.reshape(n, -1).T
        return arr if len(shape) == 1 else arr.reshape((-1,) + shape)

    data_ro = data_with_halos = data_ro_with_halos = data


class Vector:
    """Coefficient vector of a Function or an assembled 1-form (``.vec`` is the NGSolve vector)."""

    def __init__(self, vec, space=None):
        self.vec, self._space, self.bcs = vec, space, []

    def __getattr__(self, name):
        if name == "vec":
            raise AttributeError(name)
        return getattr(self.vec, name)

    def function_space(self):
        return self._space

    @property
    def dat(self):
        return Dat(self.vec, self._space.ufl_element() if self._space is not None else None)

    def _np(self):
        return self.vec.FV().NumPy()

    def get_local(self):
        return self._np().copy()

    array = get_local

    def set_local(self, values):
        self._np()[:] = values

    def __array__(self, dtype=None, copy=None):
        return np.array(self._np(), dtype=dtype)

    def __len__(self):
        return len(self._np())

    size = local_size = __len__

    def __getitem__(self, i):
        return self._np()[i]

    def __setitem__(self, i, value):
        self._np()[i] = np.asarray(value)

    def norm(self, norm_type="l2"):
        order = {"l2": 2, "l1": 1, "linf": np.inf}[norm_type.lower()]
        return float(np.linalg.norm(self._np(), order))

    def inner(self, other):
        return float(self._np() @ np.asarray(other))

    def max(self):
        return float(self._np().max())

    def min(self):
        return float(self._np().min())

    def sum(self):
        return float(self._np().sum())

    def zero(self):
        self._np()[:] = 0.0

    def axpy(self, a, x):
        self._np()[:] += a * np.asarray(x)

    def copy(self):
        v = Vector(self.vec.CreateVector(), self._space)
        v._np()[:] = self._np()
        return v

    def apply(self, mode=None):
        pass

    def _new(self, values):
        v = self.copy()
        v._np()[:] = values
        return v

    def __add__(self, other):
        return self._new(self._np() + np.asarray(other))

    def __sub__(self, other):
        return self._new(self._np() - np.asarray(other))

    def __mul__(self, a):
        return self._new(self._np() * a)

    __rmul__ = __mul__

    def __neg__(self):
        return self._new(-self._np())


class Function(ufl.Coefficient):
    """A finite element function, backed by an NGSolve GridFunction (``u.gf``)."""

    def __init__(self, V, val=None, name=None, gf=None):
        super().__init__(V)
        if isinstance(val, str):
            val, name = None, val
        self.name = _Name(name or "u")
        self.gf = gf if gf is not None else ngs.GridFunction(V.fes, name=str(self.name))
        self._field = None
        self._subs = None
        if isinstance(val, Function):
            self.gf.vec.data = val.gf.vec
        elif val is not None:
            self.gf.vec.FV().NumPy()[:] = np.asarray(val).ravel()

    def rename(self, name=None, label=None):
        if name is not None:
            self.name = _Name(name)

    def geometric_dimension(self):
        return self.function_space().mesh.dim

    @property
    def function_space(self):
        """The space; works as attribute (DOLFINx) and as method call (Firedrake)."""
        return self.ufl_function_space()

    def _ngs_field(self):
        if self._field is None:
            el = self.ufl_element()
            if isinstance(el, MixedElement):
                parts = [Part(c, e) for c, e in zip(self.gf.components, el.sub_elements)]
            else:
                parts = [Part(self.gf, el)]
            self._field = Field(parts, el)
        return self._field

    @property
    def cf(self):
        """NGSolve CoefficientFunction of the function (all components for a mixed one)."""
        fld = self._ngs_field()
        return fld.flat_value(None) if fld.mixed else self.gf

    @property
    def vec(self):
        return self.gf.vec

    def vector(self):
        return Vector(self.gf.vec, self.function_space())

    @property
    def dat(self):
        mixed = isinstance(self.ufl_element(), MixedElement)
        return Dat(self.gf.vec, self.ufl_element(), self.subfunctions if mixed else None)

    def zero(self):
        self.gf.vec[:] = 0.0
        return self

    def project(self, expr, **kwargs):
        from .solving import project
        self.gf.vec.data = project(expr, self.function_space(), **kwargs).gf.vec
        return self

    def compute_vertex_values(self, mesh=None):
        """Values at the mesh vertices, component after component."""
        m = self.function_space().mesh.ngs
        pts = np.array([list(v.point) for v in m.vertices])
        return np.asarray(self.cf(m(*pts.T))).T.ravel()

    @property
    def array(self):
        return self.gf.vec.FV().NumPy()

    def _components(self):
        if self._subs is None:
            V = self.function_space()
            self._subs = tuple(Function(V.sub(i).collapse(), name=f"{self.name}_{i}", gf=c)
                               for i, c in enumerate(self.gf.components))
            for f in self._subs:
                f._parent = self
        return self._subs

    @property
    def subfunctions(self):
        """The functions of a mixed function, sharing its data; ``(self,)`` otherwise."""
        if isinstance(self.ufl_element(), MixedElement):
            return self._components()
        return (self,)

    def sub(self, i):
        """Component i of a mixed or vector valued function."""
        comps = self._components()
        return comps[i] if comps else self

    def split(self):
        return self._components()

    def interpolate(self, expr):
        """Set the function from an expression, which may contain the function itself.

        H1, H(curl) and H(div) spaces use NGSolve's canonical interpolation (vertex values
        and moments, commuting with grad, curl and div); the others a local L2 projection.
        """
        V = self.function_space()
        cf = as_cf(expr, V.mesh, V.value_shape)
        tmp = ngs.GridFunction(V.fes)
        if not isinstance(V.ufl_element(), MixedElement):
            set_values(tmp, cf, V.ufl_element())
        else:
            fld = self._ngs_field()
            for comp, part, lo, hi in zip(tmp.components, fld.parts, fld.offsets, fld.offsets[1:]):
                shape = part.element.shape
                c = cf[lo] if not shape else ngs.CF(tuple(cf[i] for i in range(lo, hi)))
                set_values(comp, c.Reshape(shape) if len(shape) > 1 else c, part.element)
        self.gf.vec.data = tmp.vec
        return self

    def __iadd__(self, other):
        return self.assign(self + other)

    def __isub__(self, other):
        return self.assign(self - other)

    def __imul__(self, other):
        return self.assign(self * other)

    def __itruediv__(self, other):
        return self.assign(self / other)

    def assign(self, other):
        if isinstance(other, Function):
            self.gf.vec.data = other.gf.vec
        elif isinstance(other, numbers.Number) and other == 0:
            self.gf.vec[:] = 0.0
        else:
            self.interpolate(other)
        return self

    def copy(self, deepcopy=True):
        f = Function(self.function_space(), name=self.name)
        f.gf.vec.data = self.gf.vec
        return f

    def at(self, *point, **kw):
        """Value at a point ``u.at(x, y)``, or at many: ``u.at(points)`` with shape (n, dim)."""
        m = self.function_space().mesh.ngs
        if len(point) == 1 and not isinstance(point[0], numbers.Number):
            p = np.asarray(point[0], dtype=float)
            if p.ndim == 2:
                vals = np.asarray(self.cf(m(*p.T)))
                return vals[:, 0] if vals.shape[1] == 1 else vals
            point = tuple(p)
        return self.cf(m(*point))

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            return super().__call__(*args, **kwargs)
        return self.at(*args)


class ExternalFunction(ufl.Coefficient):
    """An NGSolve CoefficientFunction as a UFL coefficient, see `from_cf`."""

    def __init__(self, cf, mesh, degree):
        shape = tuple(cf.dims) if cf.dim > 1 else ()
        super().__init__(ufl.FunctionSpace(mesh.ufl_domain(), FiniteElement("DG", mesh.ufl_cell(), degree, shape=shape)))
        self.cf = cf

    def _ngs_field(self):
        return Field([Part(self.cf, self.ufl_element())], self.ufl_element())


def from_cf(cf, mesh=None, degree=1):
    """Use an NGSolve CoefficientFunction inside UFL forms.

    `degree` is the polynomial degree UFL assumes when it picks the quadrature order.
    """
    return ExternalFunction(ngs.CF(cf), mesh or default_mesh(), degree)


def material_function(mesh, values, default=0.0):
    """Piecewise constant coefficient given by material name: ``{"steel": 210e9, "air": 1}``."""
    return from_cf(mesh.ngs.MaterialCF(values, default=default), mesh, degree=0)


class _Parameters:
    """Value of a constant, kept in NGSolve Parameters."""

    def _set_value(self, value):
        arr = np.asarray(value, dtype=float)
        self._params = [ngs.Parameter(float(v)) for v in arr.flat]
        if arr.ndim == 0:
            self._cf = self._params[0]
        else:
            self._cf = ngs.CF(tuple(self._params))
            if arr.ndim > 1:
                self._cf = self._cf.Reshape(arr.shape)

    def _ngs_cf(self):
        return self._cf

    def assign(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.shape != self.ufl_shape:
            raise ValueError(f"cannot assign a value of shape {arr.shape} to a Constant of shape {self.ufl_shape}")
        for p, v in zip(self._params, arr.flat):
            p.Set(float(v))
        return self

    @property
    def value(self):
        vals = np.array([p.Get() for p in self._params]).reshape(self.ufl_shape)
        return float(vals) if not self.ufl_shape else vals

    def values(self):
        return np.array([p.Get() for p in self._params])

    def zero(self):
        return self.assign(np.zeros(self.ufl_shape))

    def __float__(self):
        return float(self.value)


class DomainConstant(_Parameters, ufl.Constant):
    """A constant tied to a mesh, ``Constant(mesh, value)``; it gives ``c * dx`` its domain."""

    def __init__(self, *args, domain=None):
        if len(args) == 2:
            domain, value = args
        else:
            value, = args
        ufl.Constant.__init__(self, domain.ufl_domain(), np.shape(value))
        self._set_value(value)


@ufl_type()
class GlobalConstant(ConstantValue, _Parameters, Counted):
    """``Constant(value)``, independent of any mesh; change it with ``assign``."""

    _ufl_noslots_ = True
    __float__ = _Parameters.__float__

    def __new__(cls, *args, domain=None):
        if len(args) == 2 or domain is not None:
            return DomainConstant(*args, domain=domain)
        return ConstantValue.__new__(cls)

    def __init__(self, value, domain=None):
        ConstantValue.__init__(self)
        Counted.__init__(self, None, GlobalConstant)
        self._shape = np.shape(value)
        self._set_value(value)

    @property
    def ufl_shape(self):
        return self._shape

    def __str__(self):
        return f"c_{self._count}"

    def __repr__(self):
        return f"Constant({self._shape!r}, {self._count!r})"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(("Constant", self._count))


Constant = GlobalConstant
for _classes in (ufl.classes.all_ufl_classes, ufl.classes.ufl_classes, ufl.classes.terminal_classes):
    _classes.add(GlobalConstant)


__all__ = ["FunctionSpace", "MixedFunctionSpace", "VectorFunctionSpace", "TensorFunctionSpace",
           "Function", "Constant", "to_cf", "as_cf", "from_cf", "material_function", "Mesh"]

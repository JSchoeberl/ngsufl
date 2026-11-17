"""Translate UFL expressions into NGSolve CoefficientFunctions.

Compound tensor operators are mapped one to one.  Index notation is collected
into lazy Einstein products and emitted as native NGSolve operations where a
pattern is recognised, as ``Einsum`` otherwise.  Derivatives of non-terminals
are expanded by UFL first, so NGSolve only ever differentiates functions.
"""

import itertools
import math
import numbers
from functools import reduce

import ngsolve as ngs
import ufl.classes as C
from ngsolve import fem
from ufl.algorithms.analysis import has_type
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.domain import extract_unique_domain

from .element import MixedElement

_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_fresh = itertools.count(-1, -1)


class NeedsDerivativeExpansion(Exception):
    pass


def _is_num(a):
    return isinstance(a, numbers.Number)


def _rank(a):
    return 0 if _is_num(a) else len(a.dims)


def _component(cf, idx):
    return cf[idx[0]] if len(idx) == 1 else cf[idx]


def _zero(shape):
    if not shape:
        return 0.0
    n = math.prod(shape)
    z = ngs.CF(tuple(0.0 for _ in range(n)))
    return z if len(shape) == 1 else z.Reshape(tuple(shape))


class Part:
    """One non-mixed block of a function: a proxy or a GridFunction."""

    def __init__(self, cf, element):
        self.cf, self.element = cf, element

    def _side(self, cf, side):
        return cf.Other() if side == "-" else cf

    def value(self, side):
        return self._side(self.cf, side)

    def grad(self, side, gdim):
        if self.element.family == "Real":
            return _zero(self.element.shape + (gdim,))
        return self._side(ngs.Grad(self.cf), side)

    def div(self, side):
        if self.element.family in ("RT", "BDM"):
            return self._side(ngs.div(self.cf), side)
        if len(self.element.shape) == 1 and self.element.family != "Real":
            return self._side(ngs.Trace(ngs.Grad(self.cf)), side)
        return None

    def curl(self, side):
        if self.element.family in ("N1curl", "N2curl"):
            return self._side(ngs.curl(self.cf), side)
        return None

    def hesse(self, side, gdim):
        shape = self.element.shape
        if self.element.family not in ("Lagrange", "DG") or len(shape) > 1:
            return None
        if shape and shape != (gdim,):
            return None
        h = self.cf.Operator("hesse")
        if shape:
            h = h.Reshape((gdim, gdim, gdim))
        return self._side(h, side)


class Field:
    """All blocks of an argument or coefficient."""

    def __init__(self, parts, element):
        self.parts, self.element = parts, element
        self.mixed = isinstance(element, MixedElement)
        self.offsets = [0]
        for p in parts:
            self.offsets.append(self.offsets[-1] + math.prod(p.element.shape))
        self._cache = {}

    def locate(self, flat):
        for p, lo, hi in zip(self.parts, self.offsets, self.offsets[1:]):
            if lo <= flat < hi:
                return p, flat - lo, hi - lo
        raise IndexError(flat)

    def flat_value(self, side):
        key = ("v", side)
        if key not in self._cache:
            comps = []
            for p in self.parts:
                v, shape = p.value(side), p.element.shape
                comps += [_component(v, i) for i in itertools.product(*map(range, shape))] if shape else [v]
            self._cache[key] = ngs.CF(tuple(comps))
        return self._cache[key]

    def flat_grad(self, side, gdim):
        key = ("g", side)
        if key not in self._cache:
            comps = []
            for p in self.parts:
                g, shape = p.grad(side, gdim), p.element.shape + (gdim,)
                comps += [_component(g, i) for i in itertools.product(*map(range, shape))]
            self._cache[key] = ngs.CF(tuple(comps), dims=(self.offsets[-1], gdim))
        return self._cache[key]


class Lazy:
    """Scalar product of tensors with labelled axes; `out` are the free labels."""

    __slots__ = ("factors", "out", "dims")

    def __init__(self, factors, out, dims):
        self.factors, self.out, self.dims = factors, tuple(out), dims

    def materialize(self, order):
        order = tuple(order)
        scalars = [cf for cf, lab in self.factors if not lab]
        tensors = [(cf, lab) for cf, lab in self.factors if lab]
        s = reduce(lambda a, b: a * b, scalars) if scalars else None
        if not tensors:
            return s
        t = _contract(tensors, order)
        return t if s is None else s * t


class Tens:
    """Tensor valued CF whose trailing axes belong to free indices."""

    __slots__ = ("cf", "free", "dims")

    def __init__(self, cf, free, dims):
        self.cf, self.free, self.dims = cf, tuple(free), dims


def _contract(factors, order):
    if len(factors) == 1:
        cf, lab = factors[0]
        if lab == order:
            return cf
        if len(lab) == 2 and lab[0] != lab[1] and order == lab[::-1]:
            return cf.trans
        if len(lab) == 2 and lab[0] == lab[1] and not order:
            return ngs.Trace(cf)
    if len(factors) == 2:
        (a, la), (b, lb) = factors
        if len(set(la)) == len(la) and len(set(lb)) == len(lb) and not (a.is_complex or b.is_complex):
            for (p, lp), (q, lq) in (((a, la), (b, lb)), ((b, lb), (a, la))):
                if len(lp) == 2 and len(lq) == 2 and lp[1] == lq[0] and order == (lp[0], lq[1]) and lp[0] != lq[1]:
                    return p * q
                if len(lp) == 2 and len(lq) == 1 and order == (lp[0],) and lp[1] == lq[0]:
                    return p * q
                if len(lp) == 2 and len(lq) == 1 and order == (lp[1],) and lp[0] == lq[0]:
                    return p.trans * q
            if not order and len(la) <= 2:
                if la == lb:
                    return ngs.InnerProduct(a, b)
                if la == lb[::-1]:
                    return ngs.InnerProduct(a, b.trans)
            if len(la) == 1 and len(lb) == 1 and la != lb:
                if order == la + lb:
                    return ngs.OuterProduct(a, b)
                if order == lb + la:
                    return ngs.OuterProduct(b, a)
    names = {}
    for _, lab in factors:
        for l in lab:
            names.setdefault(l, _LETTERS[len(names)])
    sig = ",".join("".join(names[l] for l in lab) for _, lab in factors)
    sig += "->" + "".join(names[l] for l in order)
    return fem.Einsum(sig, *[cf for cf, _ in factors])


_HANDLERS, _DISPATCH = {}, {}


def handles(*classes):
    def deco(f):
        for c in classes:
            _HANDLERS[c] = f
        return f
    return deco


def _handler(tp):
    h = _DISPATCH.get(tp)
    if h is None:
        for c in tp.__mro__:
            if c in _HANDLERS:
                h = _DISPATCH[tp] = _HANDLERS[c]
                break
        else:
            raise NotImplementedError(f"UFL node '{tp.__name__}' is not supported by ngsufl")
    return h


def _componentwise(f, a):
    dims = tuple(a.dims)
    if not dims:
        return f(a)
    comps = [f(_component(a, i)) for i in itertools.product(*map(range, dims))]
    v = ngs.CF(tuple(comps))
    return v if len(dims) == 1 else v.Reshape(dims)


def _tanh(a):
    return 1 - 2 / (ngs.exp(2 * a) + 1)


_MATH = {
    C.Sqrt: (math.sqrt, ngs.sqrt), C.Exp: (math.exp, ngs.exp), C.Ln: (math.log, ngs.log),
    C.Sin: (math.sin, ngs.sin), C.Cos: (math.cos, ngs.cos), C.Tan: (math.tan, ngs.tan),
    C.Asin: (math.asin, ngs.asin), C.Acos: (math.acos, ngs.acos), C.Atan: (math.atan, ngs.atan),
    C.Sinh: (math.sinh, ngs.sinh), C.Cosh: (math.cosh, ngs.cosh), C.Tanh: (math.tanh, _tanh),
    C.Erf: (math.erf, ngs.erf),
    C.Abs: (abs, lambda a: _componentwise(lambda c: ngs.IfPos(c, c, -c), a)),
}


class Translator:
    def __init__(self, mesh=None, arguments=None, override=None, wrap_derivative=None):
        self.mesh = mesh
        self.arguments = arguments or {}
        self.override = override or {}
        self.wrap = wrap_derivative or (lambda cf: cf)
        self.memo, self.parts = {}, {}

    # ---- driver

    def __call__(self, expr, side=None):
        r = self.tr(expr, side)
        if isinstance(r, (Lazy, Tens)):
            raise ValueError("expression has free indices")
        return r

    def tr(self, e, side=None):
        key = (id(e), side)
        hit = self.memo.get(key)
        if hit is None:
            hit = self.memo[key] = (e, _handler(type(e))(self, e, side))
        return hit[1]

    def lower(self, e, side):
        low = apply_algebra_lowering(e)
        if type(low) is type(e):
            raise NotImplementedError(f"cannot translate UFL node '{type(e).__name__}' here")
        return self.tr(low, side)

    def lazy(self, r):
        if isinstance(r, Lazy):
            return r
        if isinstance(r, Tens):
            raise ValueError("unexpected tensor valued expression in index notation")
        return Lazy([(r, ())], (), {})

    def elementwise(self, f, r):
        if isinstance(r, Lazy):
            return Lazy([(f(r.materialize(r.out)), r.out)], r.out, r.dims)
        if isinstance(r, Tens):
            return Tens(f(r.cf), r.free, r.dims)
        return f(r)

    @property
    def gdim(self):
        return self.mesh.dim

    # ---- functions

    def field(self, t):
        if isinstance(t, C.Argument):
            try:
                return self.arguments[t.number()]
            except KeyError:
                raise ValueError("expression contains a trial or test function") from None
        f = self.override.get(t)
        if f is not None:
            return f
        try:
            return t._ngs_field()
        except AttributeError:
            raise TypeError(f"coefficient {t} is not an ngsufl Function") from None

    def part(self, e):
        hit = self.parts.get(id(e))
        if hit is None:
            hit = self.parts[id(e)] = (e, self._part(e))
        return hit[1]

    def _part(self, e):
        if isinstance(e, (C.Argument, C.Coefficient)):
            f = self.field(e)
            return None if f.mixed else f.parts[0]
        if isinstance(e, C.Indexed):
            leaves = [e]
        elif isinstance(e, C.ListTensor):
            leaves, stack = [], [e]
            while stack:
                t = stack.pop()
                if isinstance(t, C.ListTensor):
                    stack.extend(reversed(t.ufl_operands))
                else:
                    leaves.append(t)
        else:
            return None
        W, flat = None, []
        for l in leaves:
            if not isinstance(l, C.Indexed):
                return None
            A, mi = l.ufl_operands
            if not isinstance(A, (C.Argument, C.Coefficient)) or len(mi) != 1 or not isinstance(mi[0], C.FixedIndex):
                return None
            if W is not None and A is not W and A != W:
                return None
            W = A
            flat.append(int(mi[0]))
        f = self.field(W)
        if not f.mixed:
            return None
        p, local, size = f.locate(flat[0])
        if local != 0 or len(flat) != size or flat != list(range(flat[0], flat[0] + size)):
            return None
        if e.ufl_shape != p.element.shape:
            return None
        return p

    @handles(C.Argument, C.Coefficient)
    def form_argument(self, e, side):
        f = self.field(e)
        return f.flat_value(side) if f.mixed else f.parts[0].value(side)

    @handles(C.Constant)
    def constant(self, e, side):
        return e._ngs_cf()

    # ---- literals and geometry

    @handles(C.ScalarValue)
    def scalar_value(self, e, side):
        return e.value()

    @handles(C.Zero)
    def zero(self, e, side):
        z = _zero(e.ufl_shape + tuple(e.ufl_index_dimensions))
        if not e.ufl_free_indices:
            return z
        dims = dict(zip(e.ufl_free_indices, e.ufl_index_dimensions))
        if e.ufl_shape:
            return Tens(z, e.ufl_free_indices, dims)
        return Lazy([(z, tuple(e.ufl_free_indices))], e.ufl_free_indices, dims)

    @handles(C.Identity)
    def identity(self, e, side):
        return ngs.Id(e.ufl_shape[0])

    @handles(C.PermutationSymbol)
    def permutation_symbol(self, e, side):
        n = e.ufl_shape[0]
        vals = []
        for idx in itertools.product(range(n), repeat=n):
            if len(set(idx)) < n:
                vals.append(0.0)
            else:
                inv = sum(1 for i in range(n) for j in range(i + 1, n) if idx[i] > idx[j])
                vals.append(-1.0 if inv % 2 else 1.0)
        return ngs.CF(tuple(vals)).Reshape(e.ufl_shape)

    @handles(C.SpatialCoordinate)
    def spatial_coordinate(self, e, side):
        return ngs.CF((ngs.x, ngs.y, ngs.z)[:e.ufl_shape[0]])

    @handles(C.FacetNormal)
    def facet_normal(self, e, side):
        n = ngs.specialcf.normal(e.ufl_shape[0])
        return -n if side == "-" else n

    @handles(C.CellDiameter, C.CellVolume, C.Circumradius, C.MinCellEdgeLength, C.MaxCellEdgeLength)
    def cell_quantity(self, e, side):
        mesh = extract_unique_domain(e).ufl_cargo()
        q = mesh.cell_quantity(type(e).__name__)
        return q.Other() if side == "-" else q

    @handles(C.Jacobian)
    def jacobian(self, e, side):
        # reference cells are NGSolve's, their vertex order differs from FEniCS
        return ngs.specialcf.JacobianMatrix(*e.ufl_shape)

    @handles(C.JacobianDeterminant)
    def jacobian_determinant(self, e, side):
        d = self.gdim
        return ngs.Det(ngs.specialcf.JacobianMatrix(d, d))

    @handles(C.JacobianInverse)
    def jacobian_inverse(self, e, side):
        d = self.gdim
        return ngs.Inv(ngs.specialcf.JacobianMatrix(d, d))

    # ---- index notation

    @handles(C.Indexed)
    def indexed(self, e, side):
        p = self.part(e)
        if p is not None:
            return p.value(side)
        A, mi = e.ufl_operands
        a = self.tr(A, side)
        cf, afree, adims = (a.cf, a.free, a.dims) if isinstance(a, Tens) else (a, (), {})
        fixed = [int(i) if isinstance(i, C.FixedIndex) else None for i in mi]
        labels = tuple(i.count() for i in mi if not isinstance(i, C.FixedIndex))
        if not labels and not afree:
            return _component(cf, tuple(fixed))
        if len(labels) < len(fixed):
            sl = tuple(slice(None) if k is None else k for k in fixed) + (slice(None),) * len(afree)
            cf = cf[sl]
        dims = dict(adims)
        for i, n in zip(mi, A.ufl_shape):
            if not isinstance(i, C.FixedIndex):
                dims[i.count()] = n
        labels += afree
        return Lazy([(cf, labels)], sorted(set(labels)), dims)

    @handles(C.ComponentTensor)
    def component_tensor(self, e, side):
        A, mi = e.ufl_operands
        a = self.lazy(self.tr(A, side))
        idx = tuple(i.count() for i in mi)
        rest = tuple(l for l in a.out if l not in idx)
        cf = a.materialize(idx + rest)
        return Tens(cf, rest, {l: a.dims[l] for l in rest}) if rest else cf

    @handles(C.IndexSum)
    def index_sum(self, e, side):
        A, mi = e.ufl_operands
        a = self.tr(A, side)
        if isinstance(a, Tens):
            shape = tuple(next(_fresh) for _ in A.ufl_shape)
            rest = tuple(l for l in a.free if l != mi[0].count())
            cf = _contract([(a.cf, shape + a.free)], shape + rest)
            return Tens(cf, rest, {l: a.dims[l] for l in rest}) if rest else cf
        a = self.lazy(a)
        i, new = mi[0].count(), next(_fresh)
        factors = [(cf, tuple(new if l == i else l for l in lab)) for cf, lab in a.factors]
        if not any(new in lab for _, lab in factors):
            factors.append((float(e.dimension()), ()))
        out = tuple(l for l in a.out if l != i)
        lz = Lazy(factors, out, a.dims)
        return lz if out else lz.materialize(())

    @handles(C.ListTensor)
    def list_tensor(self, e, side):
        p = self.part(e)
        if p is not None:
            return p.value(side)
        ops = [self.tr(o, side) for o in e.ufl_operands]
        if not any(isinstance(o, (Lazy, Tens)) for o in ops):
            v = ngs.CF(tuple(ops))
            return v if len(e.ufl_shape) == 1 else v.Reshape(e.ufl_shape)
        free = tuple(e.ufl_free_indices)
        dims = dict(zip(free, e.ufl_index_dimensions))
        cfs = [o.materialize(free) if isinstance(o, Lazy) else o.cf for o in ops]
        v = ngs.CF(tuple(cfs)).Reshape(e.ufl_shape + tuple(dims[l] for l in free))
        return Tens(v, free, dims)

    # ---- algebra

    @handles(C.Sum)
    def sum(self, e, side):
        a, b = (self.tr(o, side) for o in e.ufl_operands)
        if isinstance(a, Lazy) or isinstance(b, Lazy):
            a, b = self.lazy(a), self.lazy(b)
            order = a.out
            return Lazy([(a.materialize(order) + b.materialize(order), order)], order, {**a.dims, **b.dims})
        if isinstance(a, Tens):
            return Tens(a.cf + b.cf, a.free, a.dims)
        return a + b

    @handles(C.Product)
    def product(self, e, side):
        a, b = (self.tr(o, side) for o in e.ufl_operands)
        if not isinstance(a, Lazy) and not isinstance(b, Lazy):
            return a * b
        a, b = self.lazy(a), self.lazy(b)
        return Lazy(a.factors + b.factors, sorted(set(a.out) | set(b.out)), {**a.dims, **b.dims})

    @handles(C.Division)
    def division(self, e, side):
        a, b = (self.tr(o, side) for o in e.ufl_operands)
        if isinstance(a, Lazy):
            return Lazy(a.factors + [(1.0 / b, ())], a.out, a.dims)
        if isinstance(a, Tens):
            return Tens(a.cf / b, a.free, a.dims)
        return a / b

    @handles(C.Power)
    def power(self, e, side):
        a, b = (self.tr(o, side) for o in e.ufl_operands)
        if _is_num(a) and _is_num(b):
            return a ** b

        def f(base):
            if _is_num(b) and b == 2:
                return base * base
            return (ngs.CF(base) if _is_num(base) else base) ** b
        return self.elementwise(f, a)

    @handles(*_MATH)
    def math_function(self, e, side):
        fnum, fcf = _MATH[type(e)]
        a = self.tr(e.ufl_operands[0], side)
        return fnum(a) if _is_num(a) else self.elementwise(fcf, a)

    @handles(C.Atan2)
    def atan2(self, e, side):
        a, b = (self.tr(o, side) for o in e.ufl_operands)
        return math.atan2(a, b) if _is_num(a) and _is_num(b) else ngs.atan2(ngs.CF(a), ngs.CF(b))

    @handles(C.Conj, C.Real, C.Variable)
    def passthrough(self, e, side):
        return self.tr(e.ufl_operands[0], side)

    @handles(C.Imag)
    def imag(self, e, side):
        return _zero(e.ufl_shape)

    # ---- conditionals

    def condition(self, c, side):
        if isinstance(c, C.AndCondition):
            p, q = (self.condition(o, side) for o in c.ufl_operands)
            return lambda t, f: p(q(t, f), f)
        if isinstance(c, C.OrCondition):
            p, q = (self.condition(o, side) for o in c.ufl_operands)
            return lambda t, f: p(t, q(t, f))
        if isinstance(c, C.NotCondition):
            p = self.condition(c.ufl_operands[0], side)
            return lambda t, f: p(f, t)
        l, r = (self.tr(o, side) for o in c.ufl_operands)
        d = ngs.CF(l - r)
        if isinstance(c, C.LT):
            return lambda t, f: ngs.IfPos(-d, t, f)
        if isinstance(c, C.LE):
            return lambda t, f: ngs.IfPos(d, f, t)
        if isinstance(c, C.GT):
            return lambda t, f: ngs.IfPos(d, t, f)
        if isinstance(c, C.GE):
            return lambda t, f: ngs.IfPos(-d, f, t)
        if isinstance(c, C.EQ):
            return lambda t, f: ngs.IfPos(d, f, ngs.IfPos(-d, f, t))
        if isinstance(c, C.NE):
            return lambda t, f: ngs.IfPos(d, t, ngs.IfPos(-d, t, f))
        raise NotImplementedError(type(c).__name__)

    @handles(C.Conditional)
    def conditional(self, e, side):
        c, t, f = e.ufl_operands
        t, f = self.tr(t, side), self.tr(f, side)
        if isinstance(t, (Lazy, Tens)) or isinstance(f, (Lazy, Tens)):
            raise NotImplementedError("conditional with free indices")
        return self.condition(c, side)(t, f)

    @handles(C.MinValue, C.MaxValue)
    def minmax(self, e, side):
        a, b = (self.tr(o, side) for o in e.ufl_operands)
        if _is_num(a) and _is_num(b):
            return min(a, b) if isinstance(e, C.MinValue) else max(a, b)
        d = ngs.CF(a - b)
        return ngs.IfPos(d, b, a) if isinstance(e, C.MinValue) else ngs.IfPos(d, a, b)

    # ---- compound tensor operators

    def operands(self, e, side):
        ops = [self.tr(o, side) for o in e.ufl_operands]
        return None if any(isinstance(o, (Lazy, Tens)) for o in ops) else ops

    @handles(C.Inner)
    def inner(self, e, side):
        ops = self.operands(e, side)
        if ops is None:
            return self.lower(e, side)
        a, b = ops
        return a * b if _rank(a) == 0 else ngs.InnerProduct(a, b)

    @handles(C.Dot)
    def dot(self, e, side):
        ops = self.operands(e, side)
        if ops is not None:
            a, b = ops
            ra, rb = _rank(a), _rank(b)
            if ra == 0 or rb == 0 or (ra, rb) in ((1, 1), (2, 1), (2, 2)):
                return a * b
            if (ra, rb) == (1, 2):
                return b.trans * a
        return self.lower(e, side)

    @handles(C.Outer)
    def outer(self, e, side):
        ops = self.operands(e, side)
        if ops is not None:
            a, b = ops
            if _rank(a) == 0 or _rank(b) == 0:
                return a * b
            if _rank(a) == 1 and _rank(b) == 1:
                return ngs.OuterProduct(a, b)
        return self.lower(e, side)

    @handles(C.Cross)
    def cross(self, e, side):
        ops = self.operands(e, side)
        if ops is None:
            return self.lower(e, side)
        if e.ufl_operands[0].ufl_shape != (3,):
            raise ValueError("cross() needs 3D vectors; in 2D write a[0]*b[1] - a[1]*b[0]")
        return ngs.Cross(*ops)

    def unary(self, e, side, f):
        ops = self.operands(e, side)
        return self.lower(e, side) if ops is None else f(ops[0])

    @handles(C.Transposed)
    def transposed(self, e, side):
        return self.unary(e, side, lambda a: a.trans)

    @handles(C.Trace)
    def trace(self, e, side):
        return self.unary(e, side, ngs.Trace)

    @handles(C.Determinant)
    def determinant(self, e, side):
        return self.unary(e, side, lambda a: a if _rank(a) == 0 else ngs.Det(a))

    @handles(C.Inverse)
    def inverse(self, e, side):
        return self.unary(e, side, lambda a: 1.0 / a if _rank(a) == 0 else ngs.Inv(a))

    @handles(C.Cofactor)
    def cofactor(self, e, side):
        return self.unary(e, side, ngs.Cof)

    @handles(C.Deviatoric)
    def deviatoric(self, e, side):
        return self.unary(e, side, ngs.Deviator)

    @handles(C.Sym)
    def sym(self, e, side):
        return self.unary(e, side, ngs.Sym)

    @handles(C.Skew)
    def skew(self, e, side):
        return self.unary(e, side, ngs.Skew)

    @handles(C.Perp)
    def perp(self, e, side):
        return self.unary(e, side, lambda a: ngs.CF((-a[1], a[0])))

    @handles(C.CompoundTensorOperator, C.NablaGrad, C.NablaDiv)
    def compound(self, e, side):
        return self.lower(e, side)

    # ---- derivatives and restrictions

    @handles(C.Grad)
    def grad(self, e, side):
        f, = e.ufl_operands
        if isinstance(f, C.Restricted):
            return self.tr(C.Grad(f.ufl_operands[0]), f.side())
        if f.ufl_free_indices:
            raise NeedsDerivativeExpansion
        p = self.part(f)
        if p is not None:
            return self.wrap(p.grad(side, self.gdim))
        if isinstance(f, (C.Argument, C.Coefficient)):
            return self.wrap(self.field(f).flat_grad(side, self.gdim))
        if isinstance(f, C.Grad):
            p = self.part(f.ufl_operands[0])
            if p is not None:
                h = p.hesse(side, self.gdim)
                if h is None:
                    raise NotImplementedError(f"second derivatives of {p.element}")
                return self.wrap(h)
            if isinstance(f.ufl_operands[0], (C.Argument, C.Coefficient)):
                raise NotImplementedError("second derivatives of a mixed function, use split()")
        if isinstance(f, C.SpatialCoordinate):
            return ngs.Id(f.ufl_shape[0])
        if isinstance(f, (C.Constant, C.ConstantValue)):
            return _zero(e.ufl_shape)
        raise NeedsDerivativeExpansion

    @handles(C.Div)
    def div(self, e, side):
        f, = e.ufl_operands
        if isinstance(f, C.Restricted):
            return self.tr(C.Div(f.ufl_operands[0]), f.side())
        p = None if f.ufl_free_indices else self.part(f)
        d = p.div(side) if p is not None else None
        return self.lower(e, side) if d is None else self.wrap(d)

    @handles(C.Curl)
    def curl(self, e, side):
        f, = e.ufl_operands
        if isinstance(f, C.Restricted):
            return self.tr(C.Curl(f.ufl_operands[0]), f.side())
        p = None if f.ufl_free_indices else self.part(f)
        d = p.curl(side) if p is not None else None
        return self.lower(e, side) if d is None else self.wrap(d)

    @handles(C.CoefficientDerivative, C.VariableDerivative)
    def unexpanded_derivative(self, e, side):
        raise NeedsDerivativeExpansion

    @handles(C.PositiveRestricted)
    def positive_restricted(self, e, side):
        return self.tr(e.ufl_operands[0], "+")

    @handles(C.NegativeRestricted)
    def negative_restricted(self, e, side):
        return self.tr(e.ufl_operands[0], "-")


def expand(expr):
    """Expand derivative() and diff() nodes, leave everything else untouched."""
    if has_type(expr, C.CoefficientDerivative) or has_type(expr, C.VariableDerivative):
        return apply_derivatives(apply_algebra_lowering(expr))
    return expr


def translate(expr, mesh=None, arguments=None, override=None, wrap_derivative=None):
    """UFL expression -> NGSolve CoefficientFunction (or a number)."""
    try:
        return Translator(mesh, arguments, override, wrap_derivative)(expr)
    except NeedsDerivativeExpansion:
        expr = apply_derivatives(apply_algebra_lowering(expr))
    try:
        return Translator(mesh, arguments, override, wrap_derivative)(expr)
    except NeedsDerivativeExpansion:
        raise NotImplementedError("derivative of an expression that UFL could not expand") from None

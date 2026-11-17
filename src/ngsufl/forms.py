"""UFL forms -> NGSolve BilinearForm / LinearForm / numbers."""

import numbers

import ngsolve as ngs
from ufl.algorithms import estimate_total_polynomial_degree

from .element import MixedElement
from .translate import Field, Part, expand, translate

_FACET_TYPES = ("interior_facet",)


def mesh_of(form):
    return form.ufl_domain().ufl_cargo()


def has_interior_facets(form):
    return any(i.integral_type() in _FACET_TYPES for i in form.integrals())


def field_of(obj, element):
    if isinstance(element, MixedElement):
        return Field([Part(c, e) for c, e in zip(obj, element.sub_elements)], element)
    return Field([Part(obj, element)], element)


def _everywhere(sid):
    ids = sid if isinstance(sid, (tuple, list)) else (sid,)
    return any(s in ("everywhere", "otherwise") for s in ids)


def _degree(itg, integrand, mesh):
    deg = itg.metadata().get("quadrature_degree")
    if isinstance(deg, int):
        return deg
    try:
        deg = estimate_total_polynomial_degree(integrand)
    except Exception:
        return None
    # curved geometry: the Jacobian is a polynomial too
    return deg + mesh.dim * (mesh.curve_order - 1)


def _measure(mesh, itg, deg, ngs_order):
    """NGSolve measure; the quadrature order follows UFL's degree estimation."""
    kind, sid, kw = itg.integral_type(), itg.subdomain_id(), {}
    if _everywhere(sid):
        sid = None
    if kind == "cell":
        base, vb = ngs.dx, ngs.VOL
        if deg is not None:
            types = {el.type for el in mesh.ngs.Elements(ngs.VOL)}
            kw["intrules"] = {et: ngs.IntegrationRule(et, deg) for et in types}
    elif kind in ("exterior_facet", "interior_facet"):
        base, vb = (ngs.ds, ngs.BND) if kind == "exterior_facet" else (ngs.dx, None)
        kw["skeleton"] = True
        explicit = isinstance(itg.metadata().get("quadrature_degree"), int)
        if deg is not None and (deg > ngs_order or explicit):
            kw["bonus_intorder"] = deg - ngs_order
        if vb is None and sid is not None:
            raise NotImplementedError("subdomain ids on interior facets")
    else:
        raise NotImplementedError(f"integral type '{kind}'")
    reg = mesh.region(vb, sid) if vb is not None else None
    if reg is not None:
        kw["definedon"] = reg
    return base(**kw)


def _order(*spaces):
    degs = [V.ufl_element().embedded_superdegree for V in spaces]
    return 2 * degs[0] if len(degs) == 1 else sum(degs)


def integrals(form, arguments, override=None, ngs_order=0):
    """All integrals of a form as an NGSolve SumOfIntegrals."""
    mesh, total = mesh_of(form), None
    for itg in form.integrals():
        integrand = expand(itg.integrand())
        cf = translate(integrand, mesh, arguments, override)
        if isinstance(cf, numbers.Number):
            cf = ngs.CF(float(cf))
        term = cf * _measure(mesh, itg, _degree(itg, integrand, mesh), ngs_order)
        total = term if total is None else total + term
    return total


def _arguments(form):
    return sorted(form.arguments(), key=lambda a: a.number())


def _same(V, W):
    return V is W or V == W


def bilinear_form(form, **flags):
    test, trial = (a.ufl_function_space() for a in _arguments(form))
    dg = has_interior_facets(form)
    ftrial = trial.ngs_space(dg)
    if _same(test, trial):
        ftest = ftrial
        a = ngs.BilinearForm(ftrial, **flags)
    else:
        ftest = test.ngs_space(dg)
        a = ngs.BilinearForm(trialspace=ftrial, testspace=ftest, **flags)
    fields = {0: field_of(ftest.TestFunction(), test.ufl_element()),
              1: field_of(ftrial.TrialFunction(), trial.ufl_element())}
    a += integrals(form, fields, ngs_order=_order(test, trial))
    return a


def residual_form(F, u, **flags):
    """Linear form F(u; v) as a nonlinear NGSolve BilinearForm in u."""
    test, = (a.ufl_function_space() for a in _arguments(F))
    trial = u.function_space()
    dg = has_interior_facets(F)
    ftrial = trial.ngs_space(dg)
    if _same(test, trial):
        ftest = ftrial
        a = ngs.BilinearForm(ftrial, **flags)
    else:
        ftest = test.ngs_space(dg)
        a = ngs.BilinearForm(trialspace=ftrial, testspace=ftest, **flags)
    fields = {0: field_of(ftest.TestFunction(), test.ufl_element())}
    override = {u: field_of(ftrial.TrialFunction(), trial.ufl_element())}
    a += integrals(F, fields, override, _order(test, trial))
    return a


class _AppliedLinearForm:
    """Linear form with interior facet terms, evaluated through BilinearForm.Apply."""

    def __init__(self, form):
        test, = (a.ufl_function_space() for a in _arguments(form))
        fes = test.ngs_space(True)
        self.a = ngs.BilinearForm(fes, nonassemble=True)
        self.a += integrals(form, {0: field_of(fes.TestFunction(), test.ufl_element())}, ngs_order=_order(test))
        self.vec = self.a.mat.CreateColVector()
        self._zero = self.a.mat.CreateRowVector()
        self._zero[:] = 0.0

    def Assemble(self):
        self.a.Apply(self._zero, self.vec)
        return self


def linear_form(form):
    if has_interior_facets(form):
        return _AppliedLinearForm(form)
    test, = (a.ufl_function_space() for a in _arguments(form))
    fes = test.ngs_space()
    f = ngs.LinearForm(fes)
    f += integrals(form, {0: field_of(fes.TestFunction(), test.ufl_element())}, ngs_order=_order(test))
    return f


def _facet_functional(mesh, cf, kind, reg, deg):
    """Facet integral through a piecewise constant test function, so that normals
    and derivatives come from the adjacent volume element."""
    bonus = 0 if deg is None else deg
    if kind == "exterior_facet":
        fes = mesh.p0_space()
        kw = {"skeleton": True, "bonus_intorder": bonus}
        if reg is not None:
            kw["definedon"] = reg
        f = ngs.LinearForm(fes)
        f += cf * fes.TestFunction() * ngs.ds(**kw)
        return sum(f.Assemble().vec)
    fes = mesh.p0_space(dg=True)
    a = ngs.BilinearForm(fes, nonassemble=True)
    a += cf * fes.TestFunction() * ngs.dx(skeleton=True, bonus_intorder=bonus)
    zero, out = a.mat.CreateRowVector(), a.mat.CreateColVector()
    zero[:] = 0.0
    a.Apply(zero, out)
    return sum(out)


def functional(form):
    """Value of a form without arguments."""
    mesh = mesh_of(form)
    total = 0.0
    for itg in form.integrals():
        kind, sid = itg.integral_type(), itg.subdomain_id()
        integrand = expand(itg.integrand())
        cf = translate(integrand, mesh)
        cf = ngs.CF(float(cf)) if isinstance(cf, numbers.Number) else cf
        deg = _degree(itg, integrand, mesh)
        if kind == "cell":
            kw = {"order": 5 if deg is None else deg}
            reg = None if _everywhere(sid) else mesh.region(ngs.VOL, sid)
            if reg is not None:
                kw["definedon"] = reg
            total += ngs.Integrate(cf, mesh.ngs, **kw)
        elif kind == "exterior_facet":
            reg = None if _everywhere(sid) else mesh.region(ngs.BND, sid)
            total += _facet_functional(mesh, cf, kind, reg, deg)
        elif kind == "interior_facet":
            if not _everywhere(sid):
                raise NotImplementedError("subdomain ids on interior facets")
            total += _facet_functional(mesh, cf, kind, None, deg)
        else:
            raise NotImplementedError(f"functional over '{kind}'")
    return total

"""UFL finite elements backed by NGSolve spaces."""

import ngsolve as ngs
import ufl
from ufl import pullback as pb
from ufl import sobolevspace as sob
from ufl.finiteelement import AbstractFiniteElement

# canonical family -> (sobolev space, pullback, value kind)
_FAMILIES = {
    "Lagrange": (sob.H1, pb.identity_pullback, "scalar"),
    "DG": (sob.L2, pb.identity_pullback, "scalar"),
    "CR": (sob.L2, pb.identity_pullback, "scalar"),
    "Real": (sob.HInf, pb.identity_pullback, "scalar"),
    "N1curl": (sob.HCurl, pb.covariant_piola, "vector"),
    "N2curl": (sob.HCurl, pb.covariant_piola, "vector"),
    "RT": (sob.HDiv, pb.contravariant_piola, "vector"),
    "BDM": (sob.HDiv, pb.contravariant_piola, "vector"),
    "Regge": (sob.HEin, pb.double_covariant_piola, "matrix"),
    "HHJ": (sob.HDivDiv, pb.double_contravariant_piola, "matrix"),
}

_ALIASES = {
    "P": "Lagrange", "CG": "Lagrange", "Q": "Lagrange", "Lagrange": "Lagrange",
    "DG": "DG", "DP": "DG", "DQ": "DG", "Discontinuous Lagrange": "DG",
    "CR": "CR", "Crouzeix-Raviart": "CR",
    "R": "Real", "Real": "Real",
    "N1curl": "N1curl", "N1E": "N1curl", "Nedelec 1st kind H(curl)": "N1curl",
    "N2curl": "N2curl", "N2E": "N2curl", "Nedelec 2nd kind H(curl)": "N2curl",
    "RT": "RT", "N1F": "RT", "Raviart-Thomas": "RT",
    "BDM": "BDM", "N2F": "BDM", "Brezzi-Douglas-Marini": "BDM",
    "Regge": "Regge",
    "HHJ": "HHJ", "Hellan-Herrmann-Johnson": "HHJ",
}

_CELLS = {
    ngs.ET.SEGM: "interval", ngs.ET.TRIG: "triangle", ngs.ET.QUAD: "quadrilateral",
    ngs.ET.TET: "tetrahedron", ngs.ET.HEX: "hexahedron", ngs.ET.PRISM: "prism",
    ngs.ET.PYRAMID: "pyramid",
}


def _call(a):
    return a() if callable(a) else a


def ufl_cell(ngmesh):
    for el in ngmesh.Elements(ngs.VOL):
        return ufl.Cell(_CELLS[el.type])
    raise ValueError("empty mesh")


class FiniteElement(AbstractFiniteElement):
    """A single (non-mixed) element: family, cell, degree and block shape."""

    def __init__(self, family, cell, degree, shape=None, symmetry=False):
        if family not in _ALIASES:
            raise ValueError(f"unknown element family '{family}', known: {sorted(_ALIASES)}")
        self.family = _ALIASES[family]
        self._cell = cell if isinstance(cell, ufl.AbstractCell) else ufl.Cell(cell)
        self.degree = int(degree)
        self._sobolev, self._pullback, kind = _FAMILIES[self.family]
        tdim = _call(self._cell.topological_dimension)
        if kind == "scalar":
            self.shape = tuple(shape) if shape is not None else ()
        elif shape is not None:
            raise ValueError(f"family {self.family} does not take a shape")
        else:
            self.shape = (tdim,) if kind == "vector" else (tdim, tdim)
        self.symmetry = bool(symmetry)
        if self.symmetry:
            raise NotImplementedError("symmetric tensor elements")
        self._key = (self.family, _call(self._cell.cellname), self.degree, self.shape)

    def __repr__(self):
        return "ngsufl.FiniteElement(%r, %r, %d, shape=%r)" % self._key

    def __str__(self):
        fam, cell, deg, shape = self._key
        return f"<{fam}{deg} on {cell}{' ' + str(shape) if shape else ''}>"

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, FiniteElement) and self._key == other._key

    @property
    def sobolev_space(self):
        return self._sobolev

    @property
    def pullback(self):
        return self._pullback

    @property
    def embedded_superdegree(self):
        if self.family == "Real":
            return 0
        if self.family == "RT" or self.family == "N1curl":
            return self.degree
        if self.family in ("Regge", "HHJ"):
            return self.degree + 1
        return self.degree

    @property
    def embedded_subdegree(self):
        if self.family in ("RT", "N1curl"):
            return self.degree - 1
        return self.degree

    @property
    def cell(self):
        return self._cell

    @property
    def reference_value_shape(self):
        return self.shape

    @property
    def sub_elements(self):
        if _FAMILIES[self.family][2] == "scalar" and self.shape:
            sub = FiniteElement(self.family, self._cell, self.degree, shape=self.shape[1:])
            return [sub] * self.shape[0]
        return []


class MixedElement(AbstractFiniteElement):
    """Cartesian product of elements."""

    def __init__(self, elements):
        self._subs = list(elements)
        for e in self._subs:
            if isinstance(e, MixedElement):
                raise NotImplementedError("nested mixed elements")

    def __repr__(self):
        return "ngsufl.MixedElement([%s])" % ", ".join(repr(e) for e in self._subs)

    def __str__(self):
        return "<" + " x ".join(str(e) for e in self._subs) + ">"

    def __hash__(self):
        return hash(("mixed",) + tuple(self._subs))

    def __eq__(self, other):
        return isinstance(other, MixedElement) and self._subs == other._subs

    @property
    def sobolev_space(self):
        return sob.L2

    @property
    def pullback(self):
        return pb.MixedPullback(self)

    @property
    def embedded_superdegree(self):
        return max(e.embedded_superdegree for e in self._subs)

    @property
    def embedded_subdegree(self):
        return min(e.embedded_subdegree for e in self._subs)

    @property
    def cell(self):
        return self._subs[0].cell

    @property
    def reference_value_shape(self):
        return (sum(e.reference_value_size for e in self._subs),)

    @property
    def sub_elements(self):
        return self._subs


def make_fespace(ngmesh, element, **flags):
    """Build the NGSolve space of a UFL element."""
    if isinstance(element, MixedElement):
        subs = [make_fespace(ngmesh, e, **flags) for e in element.sub_elements]
        return ngs.FESpace(subs, dgjumps=True) if flags.get("dgjumps") else ngs.FESpace(subs)
    fam, k, shape = element.family, element.degree, element.shape
    dim = ngmesh.dim
    if fam in ("Lagrange", "DG", "CR", "Real"):
        if fam == "Lagrange":
            if k < 1:
                raise ValueError("Lagrange needs degree >= 1")
            if shape == (dim,):
                return ngs.VectorH1(ngmesh, order=k, **flags)
            base = ngs.H1(ngmesh, order=k, **flags)
        elif fam == "DG":
            if shape == (dim,):
                return ngs.VectorL2(ngmesh, order=k, **flags)
            base = ngs.L2(ngmesh, order=k, **flags)
        elif fam == "CR":
            if k != 1:
                raise ValueError("Crouzeix-Raviart needs degree 1")
            base = ngs.FESpace("nonconforming", ngmesh, order=1, **flags)
        else:
            base = ngs.NumberSpace(ngmesh, **flags)
        if not shape:
            return base
        if len(shape) == 1:
            return ngs.VectorValued(base, shape[0])
        if len(shape) == 2 and shape[0] == shape[1]:
            return ngs.MatrixValued(base, shape[0])
        raise NotImplementedError(f"block shape {shape}")
    if k < 1 and fam in ("N1curl", "N2curl", "RT", "BDM"):
        raise ValueError(f"{fam} needs degree >= 1")
    if fam == "N1curl":
        return ngs.HCurl(ngmesh, order=k, type1=True, **flags)
    if fam == "N2curl":
        return ngs.HCurl(ngmesh, order=k, **flags)
    if fam == "RT":
        return ngs.HDiv(ngmesh, order=k - 1, RT=True, **flags)
    if fam == "BDM":
        return ngs.HDiv(ngmesh, order=k, **flags)
    if fam == "Regge":
        return ngs.HCurlCurl(ngmesh, order=k, **flags)
    if fam == "HHJ":
        return ngs.HDivDiv(ngmesh, order=k, **flags)
    raise NotImplementedError(fam)

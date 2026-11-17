"""Mesh wrapper and structured mesh generators."""

import itertools

import ngsolve as ngs
import numpy as np
import ufl
from ngsolve import meshes as ngmeshes

from .element import FiniteElement, ufl_cell

_last_mesh = None
_ids = itertools.count(1000)


def default_mesh():
    if _last_mesh is None:
        raise ValueError("no mesh has been created yet, pass the mesh explicitly")
    return _last_mesh


class Mesh:
    """An NGSolve mesh that UFL accepts as a domain.

    Takes an ``ngsolve.Mesh``, a Netgen mesh, or a mesh file name.
    Subdomains and boundaries are addressed by their NGSolve names
    (``ds("left|right")``) or by 1-based index (``ds(2)``).
    """

    def __init__(self, mesh):
        global _last_mesh
        if not isinstance(mesh, ngs.Mesh):
            mesh = ngs.Mesh(mesh)
        self.ngs = mesh
        self.dim = mesh.dim
        coord = FiniteElement("Lagrange", ufl_cell(mesh), 1, shape=(self.dim,))
        self._id = next(_ids)
        self._ufl = ufl.Mesh(coord, ufl_id=self._id, cargo=self)
        self._quantities = {}
        self.curve_order = 1
        _last_mesh = self

    def ufl_id(self):
        return self._id

    def ufl_domain(self):
        return self._ufl

    def ufl_cell(self):
        return self._ufl.ufl_cell()

    def geometric_dimension(self):
        return self.dim

    @property
    def boundaries(self):
        return tuple(self.ngs.GetBoundaries())

    @property
    def materials(self):
        return tuple(self.ngs.GetMaterials())

    @property
    def num_cells(self):
        return self.ngs.ne

    @property
    def num_vertices(self):
        return self.ngs.nv

    @property
    def boundary_names(self):
        return tuple(dict.fromkeys(self.ngs.GetBoundaries()))

    def p0_space(self, dg=False):
        key = ("p0", dg)
        if key not in self._quantities:
            self._quantities[key] = ngs.L2(self.ngs, order=0, dgjumps=dg)
        return self._quantities[key]

    def boundary_ids(self, pattern):
        """1-based ids of all boundaries whose name matches the regexp."""
        mask = self.ngs.Boundaries(pattern).Mask()
        return tuple(i + 1 for i, b in enumerate(mask) if b)

    def curve(self, order):
        self.ngs.Curve(order)
        self.curve_order = order
        return self

    def region(self, vb, sid):
        if sid in ("everywhere", "otherwise", None):
            return None
        names = self.ngs.GetMaterials() if vb == ngs.VOL else self.ngs.GetBoundaries()
        get = self.ngs.Materials if vb == ngs.VOL else self.ngs.Boundaries
        reg = None
        for s in sid if isinstance(sid, (tuple, list)) else (sid,):
            if isinstance(s, str):
                r = get(s)
            else:
                if not 1 <= s <= len(names):
                    raise ValueError(f"subdomain id {s} out of range 1..{len(names)}")
                mask = ngs.BitArray(len(names))
                mask.Clear()
                mask.Set(int(s) - 1)
                r = ngs.Region(self.ngs, vb, mask)
            reg = r if reg is None else reg + r
        return reg

    def cell_quantity(self, name):
        """Piecewise constant geometric quantity, computed from the vertices."""
        if name not in self._quantities:
            fes = ngs.L2(self.ngs, order=0)
            gf = ngs.GridFunction(fes, name)
            if name == "CellVolume":
                vals = np.array(ngs.Integrate(ngs.CF(1), self.ngs, element_wise=True))
            else:
                pts = np.array([list(v.point) for v in self.ngs.vertices])
                vals = np.zeros(self.ngs.ne)
                for el in self.ngs.Elements(ngs.VOL):
                    p = pts[[v.nr for v in el.vertices]]
                    vals[el.nr] = _cell_measure(name, p)
            gf.vec.FV().NumPy()[:] = vals
            self._quantities[name] = gf
        return self._quantities[name]


def _cell_measure(name, p):
    d = [np.linalg.norm(a - b) for a, b in itertools.combinations(p, 2)]
    if name == "CellDiameter":
        return max(d)
    if name == "MinCellEdgeLength":
        return min(d)
    if name == "MaxCellEdgeLength":
        return max(d)
    if name == "Circumradius":
        if len(p) != p.shape[1] + 1:
            raise NotImplementedError("Circumradius on non-simplex cells")
        A = 2 * (p[1:] - p[0])
        b = np.sum(p[1:] ** 2 - p[0] ** 2, axis=1)
        return np.linalg.norm(np.linalg.solve(A, b) - p[0])
    raise NotImplementedError(name)


def IntervalMesh(n, a=0.0, b=1.0):
    """Interval [a, b] with boundaries 'left', 'right'."""
    return Mesh(ngmeshes.Make1DMesh(n, mapping=lambda x: a + (b - a) * x))


def UnitIntervalMesh(n):
    return IntervalMesh(n)


def RectangleMesh(nx, ny, Lx=1.0, Ly=1.0, quadrilateral=False, origin=(0.0, 0.0)):
    """Rectangle with boundaries 'bottom', 'right', 'top', 'left'."""
    ox, oy = origin
    m = ngmeshes.MakeStructured2DMesh(quads=quadrilateral, nx=nx, ny=ny,
                                      mapping=lambda x, y: (ox + Lx * x, oy + Ly * y))
    return Mesh(m)


def UnitSquareMesh(nx, ny, quadrilateral=False):
    return RectangleMesh(nx, ny, quadrilateral=quadrilateral)


def BoxMesh(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=1.0, hexahedral=False, origin=(0.0, 0.0, 0.0)):
    """Box with boundaries 'back' (x=0), 'front' (x=Lx), 'left' (y=0), 'right' (y=Ly), 'bottom', 'top'."""
    ox, oy, oz = origin
    m = ngmeshes.MakeStructured3DMesh(hexes=hexahedral, nx=nx, ny=ny, nz=nz,
                                      mapping=lambda x, y, z: (ox + Lx * x, oy + Ly * y, oz + Lz * z))
    return Mesh(m)


def UnitCubeMesh(nx, ny, nz, hexahedral=False):
    return BoxMesh(nx, ny, nz, hexahedral=hexahedral)

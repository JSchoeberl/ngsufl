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


class _Count(int):
    """An int that may also be called, ``mesh.num_cells`` and ``mesh.num_cells()``."""

    def __call__(self):
        return int(self)


class Mesh:
    """An NGSolve mesh that UFL accepts as a domain.

    Takes an ``ngsolve.Mesh``, a Netgen mesh, or a mesh file name.
    Subdomains and boundaries are addressed by their NGSolve names
    (``ds("left|right")``) or by number (``ds(2)``).  The numbers are the
    `markers` of the mesh (the built-in meshes use Firedrake's), and the
    1-based NGSolve index if the mesh has no markers.
    """

    def __init__(self, mesh, markers=None):
        global _last_mesh
        self.markers = dict(markers or {})
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

    def __call__(self):
        return self

    @property
    def geometric_dimension(self):
        return _Count(self.dim)

    topological_dimension = geometric_dimension

    @property
    def coordinates(self):
        """Vertex coordinates as a piecewise linear vector Function."""
        if "coordinates" not in self._quantities:
            from .function import Function, VectorFunctionSpace
            X = Function(VectorFunctionSpace(self, "CG", 1), name="coordinates")
            self._quantities["coordinates"] = X.interpolate(ufl.SpatialCoordinate(self._ufl))
        return self._quantities["coordinates"]

    @property
    def boundaries(self):
        return tuple(self.ngs.GetBoundaries())

    @property
    def materials(self):
        return tuple(self.ngs.GetMaterials())

    @property
    def num_cells(self):
        return _Count(self.ngs.ne)

    @property
    def num_vertices(self):
        return _Count(self.ngs.nv)

    @property
    def boundary_names(self):
        return tuple(dict.fromkeys(self.ngs.GetBoundaries()))

    def p0_space(self, dg=False):
        key = ("p0", dg)
        if key not in self._quantities:
            self._quantities[key] = ngs.L2(self.ngs, order=0, dgjumps=dg)
        return self._quantities[key]

    def boundary_ids(self, pattern):
        """Numbers of all boundaries whose name matches the regexp."""
        mask = self.ngs.Boundaries(pattern).Mask()
        if self.markers:
            hit = {n for n, b in zip(self.ngs.GetBoundaries(), mask) if b}
            return tuple(sorted(i for i, name in self.markers.items() if name in hit))
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
            elif vb == ngs.BND and self.markers:
                if s not in self.markers:
                    raise ValueError(f"boundary id {s} is not one of {sorted(self.markers)}")
                r = get(self.markers[s])
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


_PARALLEL = {"reorder", "distribution_parameters", "comm", "name", "distribution_name", "permutation_name"}


def _accept(kw):
    bad = set(kw) - _PARALLEL
    if bad:
        raise TypeError(f"unexpected keyword arguments {sorted(bad)}")


def _flip(diagonal, quadrilateral):
    if quadrilateral:
        if diagonal is not None:
            raise ValueError("Cannot specify slope of diagonal on quad meshes")
        return False
    if diagonal in (None, "left", "right"):
        return diagonal == "right"
    raise NotImplementedError(f"diagonal='{diagonal}'")


def interval(n, a, b, cls=None, **kw):
    return (cls or Mesh)(ngmeshes.Make1DMesh(n, mapping=lambda x: a + (b - a) * x), **kw)


def rectangle(nx, ny, p0, p1, quads=False, flip=False, cls=None, **kw):
    (x0, y0), (x1, y1) = p0, p1
    m = ngmeshes.MakeStructured2DMesh(quads=quads, nx=nx, ny=ny, flip_triangles=flip,
                                      mapping=lambda x, y: (x0 + (x1 - x0) * x, y0 + (y1 - y0) * y))
    return (cls or Mesh)(m, **kw)


def box(nx, ny, nz, p0, p1, hexes=False, cls=None, **kw):
    (x0, y0, z0), (x1, y1, z1) = p0, p1
    m = ngmeshes.MakeStructured3DMesh(hexes=hexes, nx=nx, ny=ny, nz=nz,
                                      mapping=lambda x, y, z: (x0 + (x1 - x0) * x, y0 + (y1 - y0) * y, z0 + (z1 - z0) * z))
    return (cls or Mesh)(m, **kw)


_M1 = {1: "left", 2: "right"}
_M2 = {1: "left", 2: "right", 3: "bottom", 4: "top"}
_M3 = {1: "back", 2: "front", 3: "left", 4: "right", 5: "bottom", 6: "top"}


def IntervalMesh(ncells, length_or_left=1.0, right=None, **kw):
    """Interval [0, length] or [left, right]; boundary 1 is the left point, 2 the right."""
    _accept(kw)
    a, b = (0.0, length_or_left) if right is None else (length_or_left, right)
    return interval(ncells, a, b, markers=_M1)


def UnitIntervalMesh(ncells, **kw):
    return IntervalMesh(ncells, 1.0, **kw)


def RectangleMesh(nx, ny, Lx=1.0, Ly=1.0, originX=0.0, originY=0.0, quadrilateral=False, diagonal=None, **kw):
    """Rectangle with corners (originX, originY) and (Lx, Ly).

    Boundaries: 1: x == originX, 2: x == Lx, 3: y == originY, 4: y == Ly,
    or by name 'left', 'right', 'bottom', 'top'.
    """
    _accept(kw)
    return rectangle(nx, ny, (originX, originY), (Lx, Ly), quadrilateral, _flip(diagonal, quadrilateral), markers=_M2)


def SquareMesh(nx, ny, L, quadrilateral=False, diagonal=None, **kw):
    return RectangleMesh(nx, ny, L, L, quadrilateral=quadrilateral, diagonal=diagonal, **kw)


def UnitSquareMesh(nx, ny, quadrilateral=False, diagonal=None, **kw):
    return RectangleMesh(nx, ny, 1.0, 1.0, quadrilateral=quadrilateral, diagonal=diagonal, **kw)


def BoxMesh(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=1.0, hexahedral=False, origin=(0.0, 0.0, 0.0), **kw):
    """Box of extent Lx, Ly, Lz.

    Boundaries: 1: x == 0, 2: x == Lx, 3: y == 0, 4: y == Ly, 5: z == 0, 6: z == Lz,
    or by name 'back', 'front', 'left', 'right', 'bottom', 'top'.
    """
    if kw.pop("diagonal", "default") != "default":
        raise NotImplementedError("diagonal of a box mesh")
    _accept(kw)
    ox, oy, oz = origin
    return box(nx, ny, nz, origin, (ox + Lx, oy + Ly, oz + Lz), hexahedral, markers=_M3)


def CubeMesh(nx, ny, nz, L, hexahedral=False, **kw):
    return BoxMesh(nx, ny, nz, L, L, L, hexahedral=hexahedral, **kw)


def UnitCubeMesh(nx, ny, nz, hexahedral=False, **kw):
    return BoxMesh(nx, ny, nz, hexahedral=hexahedral, **kw)

"""Plotting (matplotlib, NGSolve webgui) and VTK output."""

import ngsolve as ngs
import numpy as np

from .function import Function, to_cf
from .mesh import Mesh


def _lattice(et, n):
    if et == ngs.ET.TRIG:
        pts = [(i / n, j / n) for j in range(n + 1) for i in range(n + 1 - j)]
        nr = {}
        for j in range(n + 1):
            for i in range(n + 1 - j):
                nr[i, j] = len(nr)
        tris = []
        for j in range(n):
            for i in range(n - j):
                tris.append((nr[i, j], nr[i + 1, j], nr[i, j + 1]))
                if i + j < n - 1:
                    tris.append((nr[i + 1, j], nr[i + 1, j + 1], nr[i, j + 1]))
        return pts, tris
    if et == ngs.ET.QUAD:
        pts = [(i / n, j / n) for j in range(n + 1) for i in range(n + 1)]
        nr = lambda i, j: j * (n + 1) + i
        tris = []
        for j in range(n):
            for i in range(n):
                tris += [(nr(i, j), nr(i + 1, j), nr(i + 1, j + 1)), (nr(i, j), nr(i + 1, j + 1), nr(i, j + 1))]
        return pts, tris
    raise NotImplementedError(f"plot of {et} elements")


def _sample(mesh, cf, n, vb=ngs.VOL):
    m = mesh.ngs
    lat = {et: _lattice(et, n) for et in {el.type for el in m.Elements(vb)}}
    rules = {et: ngs.IntegrationRule(p, [0.0] * len(p)) for et, (p, _) in lat.items()}
    pts = m.MapToAllElements(rules, vb)
    xyz = ngs.CF((ngs.x, ngs.y, ngs.z)[:m.dim])(pts)
    vals = np.asarray(cf(pts)) if cf is not None else None
    tris, off = [], 0
    for el in m.Elements(vb):
        p, t = lat[el.type]
        tris.append(np.array(t) + off)
        off += len(p)
    return xyz, vals, np.vstack(tris)


def _scalar(vals):
    return vals[:, 0] if vals.shape[1] == 1 else np.linalg.norm(vals, axis=1)


def _plot1d(mesh, cf, ax, n, **kwargs):
    m = mesh.ngs
    ref = [(i / n,) for i in range(n + 1)]
    pts = m.MapToAllElements({ngs.ET.SEGM: ngs.IntegrationRule(ref, [0.0] * len(ref))}, ngs.VOL)
    xs = np.asarray(ngs.x(pts))[:, 0].reshape(-1, n + 1)
    vs = _scalar(np.asarray(cf(pts))).reshape(-1, n + 1)
    order = np.argsort(xs[:, 0])
    ax.plot(xs[order].ravel(), vs[order].ravel(), **kwargs)
    return ax


def _plot3d(mesh, cf, ax, n, deformation, colorbar, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    xyz, vals, tris = _sample(mesh, ngs.BoundaryFromVolumeCF(cf), n, ngs.BND)
    if deformation is not None:
        xyz = xyz + np.asarray(ngs.BoundaryFromVolumeCF(deformation)(_sample_points(mesh, n)))
    vals = _scalar(vals)
    norm = colors.Normalize(vals.min(), vals.max())
    cmap = plt.get_cmap(kwargs.pop("cmap", "viridis"))
    poly = Poly3DCollection(xyz[tris], facecolors=cmap(norm(vals[tris].mean(axis=1))),
                            edgecolors=kwargs.pop("edgecolors", "none"), **kwargs)
    ax.add_collection3d(poly)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-12))
    if colorbar:
        ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
    return ax


def _sample_points(mesh, n):
    m = mesh.ngs
    lat = {et: _lattice(et, n) for et in {el.type for el in m.Elements(ngs.BND)}}
    rules = {et: ngs.IntegrationRule(p, [0.0] * len(p)) for et, (p, _) in lat.items()}
    return m.MapToAllElements(rules, ngs.BND)


def plot(obj, ax=None, subdivision=4, colorbar=True, deformation=None, **kwargs):
    """Plot a mesh, a Function or a UFL expression with matplotlib.

    2D: filled contours; 3D: the boundary surface, optionally moved by
    `deformation` (a vector Function or expression); 1D: a curve.
    Vector fields are shown by their magnitude.  Returns the axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    if isinstance(obj, Mesh):
        if obj.dim != 2:
            raise NotImplementedError("mesh plot needs a 2D mesh")
        if ax is None:
            _, ax = plt.subplots()
        ax.set_aspect("equal")
        xy, _, tris = _sample(obj, None, 1)
        ax.triplot(Triangulation(xy[:, 0], xy[:, 1], tris), lw=kwargs.pop("lw", 0.5), color=kwargs.pop("color", "k"))
        return ax
    if isinstance(obj, Function):
        mesh, cf = obj.function_space().mesh, obj.cf
    else:
        import ufl
        mesh = ufl.domain.extract_unique_domain(obj).ufl_cargo()
        cf = to_cf(obj, mesh)
    if mesh.dim == 3:
        if deformation is not None:
            deformation = deformation.cf if isinstance(deformation, Function) else to_cf(deformation, mesh)
        return _plot3d(mesh, cf, ax, subdivision, deformation, colorbar, **kwargs)
    if ax is None:
        _, ax = plt.subplots()
    if mesh.dim == 1:
        return _plot1d(mesh, cf, ax, subdivision, **kwargs)
    ax.set_aspect("equal")
    xy, vals, tris = _sample(mesh, cf, subdivision)
    kwargs.setdefault("shading", "gouraud")
    kwargs.setdefault("cmap", "viridis")
    im = ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], tris), _scalar(vals), **kwargs)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.8)
    return ax


def Draw(obj, mesh=None, **kwargs):
    """Interactive NGSolve webgui view (needs webgui_jupyter_widgets)."""
    from ngsolve.webgui import Draw as _Draw
    if isinstance(obj, Mesh):
        return _Draw(obj.ngs, **kwargs)
    if isinstance(obj, Function):
        return _Draw(obj.cf, obj.function_space().mesh.ngs, **kwargs)
    cf = to_cf(obj, mesh)
    mesh = mesh or __import__("ufl").domain.extract_unique_domain(obj).ufl_cargo()
    return _Draw(cf, mesh.ngs, **kwargs)


class VTKFile:
    """``VTKFile("out").write(u, p, time=t)`` writes VTK files through NGSolve."""

    def __init__(self, filename, subdivision=1):
        self.filename = filename[:-4] if filename.endswith((".pvd", ".vtk", ".vtu")) else filename
        self.subdivision = subdivision
        self._out = None

    def write(self, *functions, time=None):
        if self._out is None:
            mesh = functions[0].function_space().mesh
            self._out = ngs.VTKOutput(mesh.ngs, coefs=[f.cf for f in functions],
                                      names=[f.name for f in functions],
                                      filename=self.filename, subdivision=self.subdivision)
        self._out.Do() if time is None else self._out.Do(time=time)

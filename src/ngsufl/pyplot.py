"""Matplotlib plots with the names and call signatures of ``firedrake.pyplot``."""

import ngsolve as ngs
import numpy as np

from .function import Function, to_cf
from .mesh import Mesh
from .output import _sample, _scalar

__all__ = ["plot", "triplot", "tricontourf", "tricontour", "trisurf", "tripcolor", "quiver"]


def _field(function):
    if isinstance(function, Function):
        return function.function_space().mesh, function.cf
    import ufl
    mesh = ufl.domain.extract_unique_domain(function).ufl_cargo()
    return mesh, to_cf(function, mesh)


def _axes(kwargs, projection=None):
    import matplotlib.pyplot as plt
    axes = kwargs.pop("axes", None)
    if axes is None:
        axes = plt.figure().add_subplot(111, projection=projection)
    return axes


def _triangulation(function, kwargs):
    from matplotlib.tri import Triangulation
    mesh, cf = _field(function)
    if mesh.dim != 2:
        raise NotImplementedError("this plot needs a 2D mesh")
    xy, vals, tris = _sample(mesh, cf, kwargs.pop("num_sample_points", 4))
    return Triangulation(xy[:, 0], xy[:, 1], tris), _scalar(vals)


def _plot_2d(method, function, args, kwargs):
    kwargs.pop("complex_component", None)
    axes = _axes(kwargs)
    tri, vals = _triangulation(function, kwargs)
    artist = getattr(axes, method)(tri, vals, *args, **kwargs)
    axes.autoscale_view()
    return artist


def tripcolor(function, *args, **kwargs):
    """Pseudo-colour plot of a 2D function (the magnitude of a vector field)."""
    kwargs.setdefault("shading", "gouraud")
    return _plot_2d("tripcolor", function, args, kwargs)


def tricontourf(function, *args, **kwargs):
    return _plot_2d("tricontourf", function, args, kwargs)


def tricontour(function, *args, **kwargs):
    return _plot_2d("tricontour", function, args, kwargs)


def trisurf(function, *args, **kwargs):
    """3D surface plot of a 2D function."""
    kwargs.pop("complex_component", None)
    axes = _axes(kwargs, projection="3d")
    tri, vals = _triangulation(function, kwargs)
    return axes.plot_trisurf(tri, vals, *args, **kwargs)


def triplot(mesh, axes=None, interior_kw={}, boundary_kw={}):
    """Plot a 2D mesh; the boundary parts are coloured and labelled, show them with ``axes.legend()``."""
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation
    if not isinstance(mesh, Mesh) or mesh.dim != 2:
        raise NotImplementedError("triplot needs a 2D mesh")
    axes = _axes({"axes": axes})
    m = mesh.ngs
    pts = np.array([list(v.point) for v in m.vertices])
    xy, _, tris = _sample(mesh, None, 1)
    kw = {"linewidth": 0.5, "color": "k"}
    kw.update(interior_kw)
    lines = axes.triplot(Triangulation(xy[:, 0], xy[:, 1], tris), **kw)
    label = {name: i for i, name in mesh.markers.items()}
    segments = {}
    for el in m.Elements(ngs.BND):
        segments.setdefault(el.mat, []).append(pts[[v.nr for v in el.vertices]])
    cycle = boundary_kw.get("colors") or [f"C{i}" for i in range(10)]
    out = list(lines)
    for i, (name, segs) in enumerate(sorted(segments.items(), key=lambda s: label.get(s[0], s[0]))):
        kw = {"linewidth": 2.0, "colors": cycle[i % len(cycle)], "label": label.get(name, name)}
        kw.update({k: v for k, v in boundary_kw.items() if k != "colors"})
        out.append(axes.add_collection(LineCollection(segs, **kw)))
    axes.autoscale_view()
    return out


def quiver(function, **kwargs):
    """Arrows of a 2D vector field at the mesh vertices, coloured by magnitude."""
    kwargs.pop("complex_component", None)
    axes = _axes(kwargs)
    mesh, cf = _field(function)
    if mesh.dim != 2:
        raise NotImplementedError("quiver needs a 2D mesh")
    if cf.dim != 2:
        raise ValueError("quiver needs a vector field")
    pts = np.array([list(v.point) for v in mesh.ngs.vertices])
    vals = np.asarray(cf(mesh.ngs(*pts.T)))
    return axes.quiver(pts[:, 0], pts[:, 1], vals[:, 0], vals[:, 1], np.linalg.norm(vals, axis=1), **kwargs)


def plot(function, *args, num_sample_points=10, **kwargs):
    """Plot a 1D function."""
    kwargs.pop("complex_component", None)
    axes = _axes(kwargs)
    mesh, cf = _field(function)
    if mesh.dim != 1:
        raise NotImplementedError("plot needs a 1D mesh, use tripcolor or tricontourf in 2D")
    n, m = num_sample_points, mesh.ngs
    ref = [(i / n,) for i in range(n + 1)]
    mp = m.MapToAllElements({ngs.ET.SEGM: ngs.IntegrationRule(ref, [0.0] * len(ref))}, ngs.VOL)
    xs = np.asarray(ngs.x(mp))[:, 0].reshape(-1, n + 1)
    vs = _scalar(np.asarray(cf(mp))).reshape(-1, n + 1)
    order = np.argsort(xs[:, 0])
    return axes.plot(xs[order].ravel(), vs[order].ravel(), *args, **kwargs)

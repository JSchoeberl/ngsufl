"""Dirichlet boundary conditions."""

import inspect

import ngsolve as ngs
import numpy as np

from .function import as_cf, set_values


class DirichletBC:
    """``DirichletBC(V, g, where)``.

    `where` is "on_boundary", a boundary name / regexp ("left|right"), a 1-based
    boundary id or tuple of ids, or a callable marking boundary points:
    ``lambda x: np.isclose(x[0], 0)`` (vectorised) or ``lambda x, on_boundary: ...``.
    `V` may be a component, ``W.sub(0)`` or ``V.sub(1)``.
    """

    def __init__(self, V, g, where="on_boundary"):
        self.space, self.g, self.where = V, g, where
        self.mesh = V.mesh
        self._dofs = None
        self._pins = {}
        if not callable(where):
            self._region()

    def _marked(self):
        """Marked vertices and the boundary elements all of whose vertices are marked."""
        m = self.mesh.ngs
        pts = np.array([list(v.point) for v in m.vertices])
        on_bnd = np.zeros(len(pts), dtype=bool)
        for el in m.Elements(ngs.BND):
            on_bnd[[v.nr for v in el.vertices]] = True
        if len(inspect.signature(self.where).parameters) >= 2:
            ok = np.array([bool(self.where(p, bool(b))) for p, b in zip(pts, on_bnd)])
        else:
            X = np.zeros((3, len(pts)))
            X[:pts.shape[1]] = pts.T
            ok = np.broadcast_to(np.asarray(self.where(X), dtype=bool), (len(pts),))
        marked = np.zeros(m.GetNE(ngs.BND), dtype=bool)
        for el in m.Elements(ngs.BND):
            marked[el.nr] = all(ok[v.nr] for v in el.vertices)
        return pts, ok, marked

    def _region(self):
        if self.where == "on_boundary":
            return self.mesh.ngs.Boundaries(".*")
        return self.mesh.region(ngs.BND, self.where)

    def _local_dofs(self):
        fes = self.space.fes
        if callable(self.where):
            pts, ok, marked = self._marked()
            ba = ngs.BitArray(fes.ndof)
            ba.Clear()
            for el in fes.Elements(ngs.BND):
                if marked[el.nr]:
                    for d in el.dofs:
                        if d >= 0:
                            ba.Set(d)
            # single points: vertex dofs hold point values
            for nr in np.nonzero(ok)[0]:
                dofs = [d for d in fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, int(nr))) if d >= 0]
                if dofs and not all(ba[d] for d in dofs):
                    self._pins[int(nr)] = (dofs, pts[nr])
                    for d in dofs:
                        ba.Set(d)
            return ba
        return fes.GetDofs(self._region())

    def dofs(self):
        """BitArray of constrained dofs of the root space."""
        if self._dofs is None:
            local = self._local_dofs()
            rng = self.space.dof_range()
            ba = ngs.BitArray(self.space.root.fes.ndof)
            ba.Clear()
            for i, b in enumerate(local):
                if b:
                    ba.Set(rng.start + i)
            self._dofs = (local, ba)
        return self._dofs[1]

    def function_space(self):
        return self.space

    @property
    def sub_domain(self):
        return self.where

    def apply(self, *targets):
        """Write the boundary values into Functions and vectors, and attach the condition to matrices."""
        for t in targets:
            if isinstance(t, ngs.GridFunction):
                self._set(t)
            elif hasattr(t, "gf"):
                self._set(t.gf)
            elif hasattr(t, "mat"):
                t.bcs.append(self)
            else:
                gf = ngs.GridFunction(self.space.root.fes)
                gf.vec.data = t.vec
                self._set(gf)
                t.vec.data = gf.vec
                t.bcs.append(self)

    def _set(self, gf):
        self.dofs()
        local = np.array(list(self._dofs[0]), dtype=bool)
        fes = self.space.fes
        tmp = ngs.GridFunction(fes)
        cf = as_cf(self.g, self.mesh, tuple(tmp.dims) if tmp.dim > 1 else ())
        region = self.mesh.ngs.Boundaries(".*") if callable(self.where) else self._region()
        set_values(tmp, cf, self.space.ufl_element(), region)
        start = self.space.dof_range().start
        vals = gf.vec.FV().NumPy()
        idx = np.nonzero(local)[0]
        vals[start + idx] = tmp.vec.FV().NumPy()[idx]
        for dofs, pt in self._pins.values():
            v = np.atleast_1d(cf(self.mesh.ngs(*pt)))
            if len(v) == len(dofs):
                vals[start + np.array(dofs)] = v


def free_dofs(V, bcs):
    free = ngs.BitArray(V.fes.FreeDofs())
    for bc in bcs:
        free &= ~bc.dofs()
    return free

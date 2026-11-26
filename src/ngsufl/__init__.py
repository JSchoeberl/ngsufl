"""ngsufl: the Unified Form Language on the NGSolve finite element core.

``from ngsufl import *`` gives the UFL language and a front end in the style of
Firedrake (meshes, spaces, functions, boundary conditions, solve).
``from ngsufl.fenics import *`` gives the same in the style of legacy FEniCS.
"""

from ufl import *  # noqa: F401,F403
import ufl as _ufl


def _check_ngsolve():
    import re
    import warnings
    import ngsolve
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:-(\d+))?", ngsolve.__version__)
    if m and tuple(int(g or 0) for g in m.groups()) < (6, 2, 2608, 400):
        warnings.warn(f"ngsufl needs an NGSolve newer than the release 6.2.2608, found {ngsolve.__version__}; "
                      "install one with: pip install --pre --upgrade ngsolve", stacklevel=3)


_check_ngsolve()

from .bcs import DirichletBC
from .element import FiniteElement, MixedElement, TensorElement, VectorElement
from .function import (Constant, Function, FunctionSpace, MixedFunctionSpace,
                       TensorFunctionSpace, TestFunction, TestFunctions, TrialFunction,
                       TrialFunctions, Vector, VectorFunctionSpace, as_cf, from_cf,
                       material_function, to_cf)
from .mesh import (BoxMesh, CubeMesh, IntervalMesh, Mesh, RectangleMesh, SquareMesh,
                   UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh)
from .output import Draw, VTKFile, plot
from .solving import (ConvergenceError, LinearSolver, LinearVariationalProblem,
                      LinearVariationalSolver, Matrix, MixedVectorSpaceBasis,
                      NonlinearVariationalProblem, NonlinearVariationalSolver,
                      VectorSpaceBasis, assemble, errornorm, interpolate, norm, project, solve)
from .translate import translate

__version__ = "0.2.0"

COMM_WORLD = COMM_SELF = None
Cofunction = Vector


def CellSize(mesh):
    return _ufl.CellDiameter(mesh)


_own = ["DirichletBC", "FiniteElement", "MixedElement", "VectorElement", "TensorElement",
        "Constant", "Function", "FunctionSpace", "MixedFunctionSpace", "TensorFunctionSpace",
        "VectorFunctionSpace", "Vector", "Cofunction", "Matrix", "as_cf", "to_cf", "from_cf",
        "material_function", "BoxMesh", "CubeMesh", "IntervalMesh", "Mesh", "RectangleMesh",
        "SquareMesh", "UnitCubeMesh", "UnitIntervalMesh", "UnitSquareMesh", "Draw", "VTKFile",
        "plot", "ConvergenceError", "LinearSolver", "LinearVariationalProblem",
        "LinearVariationalSolver", "NonlinearVariationalProblem", "NonlinearVariationalSolver",
        "VectorSpaceBasis", "MixedVectorSpaceBasis", "assemble", "errornorm", "interpolate",
        "norm", "project", "solve", "translate", "CellSize", "COMM_WORLD", "COMM_SELF"]
__all__ = sorted(set(_ufl.__all__) | set(_own))

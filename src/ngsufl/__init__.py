"""ngsufl: the Unified Form Language on the NGSolve finite element core.

``from ngsufl import *`` gives the UFL language plus a small front end
(meshes, spaces, functions, boundary conditions, solve).
"""

from ufl import *  # noqa: F401,F403
import ufl as _ufl

from .bcs import DirichletBC
from .element import FiniteElement, MixedElement
from .function import (Constant, Function, FunctionSpace, MixedFunctionSpace,
                       TensorFunctionSpace, VectorFunctionSpace, as_cf, from_cf,
                       material_function, to_cf)
from .mesh import (BoxMesh, IntervalMesh, Mesh, RectangleMesh, UnitCubeMesh,
                   UnitIntervalMesh, UnitSquareMesh)
from .output import Draw, VTKFile, plot
from .solving import (LinearVariationalProblem, LinearVariationalSolver,
                      NonlinearVariationalProblem, NonlinearVariationalSolver,
                      assemble, errornorm, interpolate, norm, project, solve)
from .translate import translate

__version__ = "0.1.0"

_own = ["DirichletBC", "FiniteElement", "MixedElement", "Constant", "Function",
        "FunctionSpace", "MixedFunctionSpace", "TensorFunctionSpace",
        "VectorFunctionSpace", "as_cf", "to_cf", "from_cf", "material_function", "BoxMesh", "IntervalMesh", "Mesh",
        "RectangleMesh", "UnitCubeMesh", "UnitIntervalMesh", "UnitSquareMesh",
        "Draw", "VTKFile", "plot", "LinearVariationalProblem",
        "LinearVariationalSolver", "NonlinearVariationalProblem",
        "NonlinearVariationalSolver", "assemble", "errornorm", "interpolate",
        "norm", "project", "solve", "translate"]
__all__ = sorted(set(_ufl.__all__) | set(_own))

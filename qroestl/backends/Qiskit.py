import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from typing import Generic, TypeVar, Optional, List
from dataclasses import dataclass, field
from functools import reduce
import qiskit
from dwave.plugins.qiskit import DWaveMinimumEigensolver
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.opflow.operator_base import OperatorBase
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.utils import QuantumInstance
import qiskit_optimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qroestl.utils import Utils
from qroestl.model import Solver, Solution, QuboConvertible



TCandidate = TypeVar('Solution Candidate', bound='Solution')
TProblem = TypeVar('Problem', bound='Problem')


@dataclass
class QiskitSolver(Generic[TCandidate, TProblem], Solver[TCandidate, TProblem]):

    quantum_instance: Optional[QuantumInstance] = None
    kwargs: "kwargs" = field(default_factory=lambda: {})

    def solve_(self, p: TProblem, s=Solution[TCandidate, TProblem]()) -> Solution[TCandidate, TProblem]:
        if isinstance(p, QuboConvertible):
            p_converted = p.to_qubo()
        pre = reduce(lambda x, f: f(x), self.pre, p_converted)
        post = reduce(lambda x, f: f(x), self.post, self.run(pre))
        return s.eval(p, Utils.bits2idx(len(p.S))(post))


class CanHandleOpsAndQPs:
    def run(self, obj) -> List[int]:
        if isinstance(obj, OperatorBase):
            return sample_most_likely(self.algo.compute_minimum_eigenvalue(obj).eigenstate)
        elif isinstance(obj, qiskit_optimization.QuadraticProgram) or isinstance(obj, qiskit.optimization.QuadraticProgram):
            return self.mes.solve(obj).x


@dataclass
class QAOA(Generic[TCandidate, TProblem], QiskitSolver[TCandidate, TProblem], CanHandleOpsAndQPs):
    name: str = "Qiskit-QAOA"

    def __post_init__(self) -> None:
        self.algo = qiskit.algorithms.QAOA(**self.kwargs, quantum_instance=self.quantum_instance)
        self.mes = MinimumEigenOptimizer(self.algo)


@dataclass
class VQE(Generic[TCandidate, TProblem], QiskitSolver[TCandidate, TProblem], CanHandleOpsAndQPs):
    name: str = "Qiskit-VQE"

    def __post_init__(self) -> None:
        self.algo = qiskit.algorithms.VQE(**self.kwargs, quantum_instance=self.quantum_instance)
        self.mes = MinimumEigenOptimizer(self.algo)


@dataclass
class NumpyExact(Generic[TCandidate, TProblem], QiskitSolver[TCandidate, TProblem], CanHandleOpsAndQPs):
    name: str = "Qiskit-NumpyExact"

    def __post_init__(self) -> None:
        self.algo = NumPyMinimumEigensolver(**self.kwargs)
        self.mes = MinimumEigenOptimizer(self.algo)


@dataclass
class CPLEX(Generic[TCandidate, TProblem], QiskitSolver[TCandidate, TProblem], CanHandleOpsAndQPs):
    name: str = "Qiskit-CPLEX"

    def __post_init__(self) -> None:
        self.algo = CplexOptimizer()
        self.mes = self.algo

@dataclass
class DWaveAnnealer(Generic[TCandidate, TProblem], QiskitSolver[TCandidate, TProblem], CanHandleOpsAndQPs):
    name: str = "Qiskit-DWaveAnnealer"

    def __post_init__(self) -> None:
        self.pre += [Utils.convert_qubo_to_legacy]
        self.algo = DWaveMinimumEigensolver(**self.kwargs)
        from qiskit.optimization.algorithms import MinimumEigenOptimizer
        self.mes = MinimumEigenOptimizer(self.algo)

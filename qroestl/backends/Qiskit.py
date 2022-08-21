import warnings
from abc import abstractmethod

import numpy as np
from qiskit import BasicAer
from qiskit.providers.ibmq import IBMQ
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from run import BackendConfig

warnings.filterwarnings('ignore', category=DeprecationWarning)

from typing import Generic, TypeVar
from dataclasses import dataclass
import qiskit
from dwave.plugins.qiskit import DWaveMinimumEigensolver
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer, GurobiOptimizer
from qroestl.utils import Utils
from qroestl.model import Optimizer, Converter

TCandidate = TypeVar('Solution Candidate', bound='Solution')
TProblem = TypeVar('Problem', bound='Problem')


STATEVECTOR_SIM = QuantumInstance(BasicAer.get_backend('statevector_simulator'))


class QiskitQuboConvertible(Generic[TCandidate, TProblem]):
    def to_qiskit_qubo(self, p: TProblem) -> QuadraticProgram:
        return QuadraticProgramToQubo().convert(self.to_qiskit_qp(p))


class QiskitQPConvertible(Generic[TCandidate, TProblem]):
    @abstractmethod
    def to_qiskit_qp(self, p: TProblem) -> QuadraticProgram:
        raise NotImplementedError


class QiskitOperatorConvertible(Generic[TCandidate, TProblem]):
    def to_qiskit_op(self, p: TProblem) -> "Operator":
        return self.to_qiskit_qubo(p).to_ising()[0]


class QUBOConverter(Converter):
    def convert(self, p, a):
        if isinstance(a, QiskitQuboConvertible):
            return a.to_qiskit_qubo(p)
        else:
            raise ValueError('Approach not convertible to Qiskit QUBO')


@dataclass
class QiskitOptimizer(Generic[TCandidate, TProblem], Optimizer[TCandidate, TProblem]):
    optimizer: "Qiskit-Optimizer" = None
    name: str = 'Qiskit'
    converter: Converter = QUBOConverter()

    def optimize_(self, p, p_conv, a, s):
        res = self.optimizer.solve(p_conv).x
        print("Solver res=", res)
        return s.eval(p, Utils.bits2idx(len(p.S))(np.clip(np.rint(res), 0, 1))), None


@dataclass
class QAOA(Generic[TCandidate, TProblem], QiskitOptimizer[TCandidate, TProblem]):
    qdev: 'Dev' = None

    def __post_init__(self) -> None:
        self.name: str = self.name + "-QAOA"
        #self.qdev = IBMQ.load_account().get_backend('ibmq_brooklyn')
        self.qdev = BackendConfig.qdev
        self.algo = qiskit.algorithms.QAOA(**self.kwargs, quantum_instance=self.qdev)
        self.optimizer = MinimumEigenOptimizer(self.algo)


@dataclass
class VQE(Generic[TCandidate, TProblem], QiskitOptimizer[TCandidate, TProblem]):
    def __post_init__(self) -> None:
        self.name: str = self.name + "-VQE"
        self.algo = qiskit.algorithms.VQE(**self.kwargs, quantum_instance=self.qdev)
        self.optimizer = MinimumEigenOptimizer(self.algo)


@dataclass
class NumpyExact(Generic[TCandidate, TProblem], QiskitOptimizer[TCandidate, TProblem]):
    def __post_init__(self) -> None:
        self.name: str = self.name + "-NumpyExact"
        self.optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver(**self.kwargs))


@dataclass
class CPLEX(Generic[TCandidate, TProblem], QiskitOptimizer[TCandidate, TProblem]):
    def __post_init__(self) -> None:
        self.name: str = self.name + "-CPLEX"
        self.optimizer = CplexOptimizer()

@dataclass
class Gurobi(Generic[TCandidate, TProblem], QiskitOptimizer[TCandidate, TProblem]):
    def __post_init__(self) -> None:
        self.name: str = self.name + "-Gurobi"
        self.optimizer = GurobiOptimizer()


@dataclass
class DWaveAnnealer(Generic[TCandidate, TProblem], QiskitOptimizer[TCandidate, TProblem]):
    def __post_init__(self) -> None:
        self.name: str = self.name + "-DWave"
        self.pre += [Utils.convert_qubo_to_legacy]
        self.algo = DWaveMinimumEigensolver(**self.kwargs)
        from qiskit.optimization.algorithms import MinimumEigenOptimizer
        self.mes = MinimumEigenOptimizer(self.algo)



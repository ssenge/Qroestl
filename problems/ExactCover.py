import warnings

from problems import SetCover

warnings.filterwarnings('ignore', category=DeprecationWarning)

from dataclasses import dataclass
from qiskit_optimization.applications import ExactCover
from qiskit_optimization import QuadraticProgram


# Similar to SetCover but no duplicate elements in sets ("lists") allowed

@dataclass
class Problem(SetCover.Problem):
    def to_qp(self) -> QuadraticProgram:
        return ExactCover(self.S).to_quadratic_program()


@dataclass
class Greedy(SetCover.Greedy):
    pass

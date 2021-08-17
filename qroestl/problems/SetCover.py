import warnings

from qroestl.model import Model

warnings.filterwarnings('ignore', category=DeprecationWarning)

from dataclasses import dataclass
from typing import List
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from qroestl.utils import Utils
from qroestl.model.Model import Solver, Solution, QPConvertible, QuboConvertible, OperatorConvertible


# https://en.wikipedia.org/wiki/Set_cover_problem


TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List]


@dataclass
class Problem(Model.Problem[TCandidate], QPConvertible, QuboConvertible, OperatorConvertible):
    U: List[int] = None
    S: List[List[int]] = None

    def __post_init__(self) -> None:
        self.U = np.array(self.U)
        self.S = np.array(self.S+[[]], dtype=object)[:-1]

    def feasible_equal(self, c: TCandidate) -> bool:
        return np.array_equal(Utils.union(self.S, c), self.U)

    def feasible_superset(self, c: TCandidate) -> bool:
        return np.all(np.isin(self.U, Utils.union(self.S, c)))

    feasible = feasible_equal

    def cost(self, c: TCandidate) -> bool:
        return len(c)

    def all_solutions(self) -> List[TCandidate]:
        return Utils.powerset(range(len(self.S)))


    def to_qp(self) -> QuadraticProgram:
        qp = QuadraticProgram('Set Cover')
        qp.binary_var_list(len(self.S), name='s')
        qp.minimize(linear=[1]*len(self.S))  # add weights, TODO: currently only const = 1
        for u in self.U:  # add cover constraints
            qp.linear_constraint(linear={f's{i}': 1 for i, s in enumerate(self.S) if u in s},
                                  sense='>=', rhs=1, name=f'cover u{u}')
        #print(qp.export_as_lp_string())
        return qp

    def to_qubo(self) -> QuadraticProgram:
        return QuadraticProgramToQubo().convert(self.to_qp())

    def to_op(self) -> "Operator":
        return self.to_qubo().to_ising()[0]


class Generator:
    @classmethod
    def linear(cls, U_n: int = 10, S_n: int = 5) -> Problem:
        return Problem(U := list(range(U_n)), [U[i:i + S_n] for i in range(0, len(U), S_n)])


@dataclass
class Greedy(Solver[TCandidate, Problem]):
    name: str = "Greedy"

    def solve_(self, p: Problem, s=Solution[TCandidate, Problem]()) -> Solution[TCandidate, Problem]:
        def rec(pos_in_cur_S=-1, U=p.U, S=p.S, sol=[]):
            return sol if U.size == 0 else \
                rec(pos_in_cur_S := np.argmax([len(np.intersect1d(U, s)) for s in S]),
                    np.setdiff1d(U, max_S := S[pos_in_cur_S]),
                    np.delete(S, pos_in_cur_S),
                    sol + [Utils.where(p.S, max_S)])

        return s.eval(p, rec())

from dataclasses import dataclass
from typing import List
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from model import Model
from model.Model import Solver, Solution, QPConvertible, QuboConvertible, OperatorConvertible
from utils import Utils


# https://en.wikipedia.org/wiki/Maximum_coverage_problem


TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List]


@dataclass
class Problem(Model.Problem[TCandidate], QPConvertible, QuboConvertible, OperatorConvertible):
    k: int = 1
    S: List[List[int]] = None

    def __post_init__(self) -> None:
        self.S = np.array(self.S+[[]], dtype=object)
        self.S = self.S[:-1]
        self.U = Utils.union(self.S)

    def feasible(self, c: TCandidate) -> bool:
        return len(c) <= self.k

    def cost(self, c: TCandidate) -> bool:
        return len(Utils.union(self.S, c))

    def all_solutions(self) -> List[TCandidate]:
        return Utils.powerset(range(len(self.S)))

    def to_qp(self) -> QuadraticProgram:
        qp = QuadraticProgram('Maximum Set Coverage')
        qp.binary_var_list(len(self.S), name='s')
        qp.binary_var_list(list(self.U), name='u')
        qp.maximize(linear=([0]*len(self.S))+([1]*len(self.U)))  # add weights, TODO: currently only const = 1
        qp.linear_constraint(linear={f's{i}': 1 for i in range(len(self.S))}, sense='<=', rhs=self.k, name='max k sets')
        for u in self.U:  # add cover constraints
            qp.linear_constraint(linear={**{f's{i}': 1 for i in range(len(self.S)) if u in self.S[i]}, **{f'u{u}': -1}},
                                 sense='>=', rhs=0, name=f'cover y{u}')
        #print(qp.export_as_lp_string())
        return qp

    def to_qubo(self) -> QuadraticProgram:
        return QuadraticProgramToQubo().convert(self.to_qp())

    def to_op(self) -> "Operator":
        return self.to_qubo().to_ising()[0]


@dataclass
class Greedy(Solver[TCandidate, Problem]):
    name: str = "Greedy"

    def solve_(self, p: Problem, s=Solution[TCandidate, Problem]()) -> Solution[TCandidate, Problem]:

        def rec(max_s=-1, i=min(len(p.S), p.k), U=p.U, S=p.S, sol=[]) -> TCandidate:
            return sol if i == 0 else rec(
                max_s:=np.argmax([len(np.intersect1d(s, U)) for s in S]),
                i-1,
                np.setdiff1d(U, S[max_s]),
                np.delete(S, max_s),
                sol + [Utils.where(p.S, S[max_s])])

        return s.eval(p, rec())






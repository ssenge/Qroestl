from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from qroestl.model import Model
from qroestl.model.Model import Solver, Solution, QPConvertible, QuboConvertible, OperatorConvertible
from qroestl.utils import Utils


# https://en.wikipedia.org/wiki/Maximum_coverage_problem


TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List[int]]


@dataclass
class Problem(Model.Problem[TCandidate], QPConvertible, QuboConvertible, OperatorConvertible):
    name: str = 'Set Coverage'
    k: int = 1
    S: Optional[List[List[int]]] = None
    W: Optional[List[int]] = None

    def __post_init__(self) -> None:
        self.S = np.array(self.S+[[]], dtype=object)
        self.S = self.S[:-1]
        self.U = Utils.union_non_sorted(self.S)
        self.W = self.W if self.W is not None else [1]*len(self.U)
        self.SW = {s: w for s, w in zip(self.U, self.W)}

    def feasible(self, c: TCandidate) -> bool:
        return len(c) <= self.k

    def cost(self, c: TCandidate) -> float:
        return sum([self.SW[s] for s in Utils.union(self.S, c)])

    def all_solutions(self) -> List[TCandidate]:
        return Utils.powerset(range(len(self.S)))

    def to_qp(self) -> QuadraticProgram:
        qp = QuadraticProgram('Maximum Set Coverage')
        qp.binary_var_list(len(self.S), name='s')
        qp.binary_var_list(list(self.U), name='u')
        qp.maximize(linear=([0]*len(self.S))+self.W)
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

        def rec(max_s=-1, i=min(len(p.S), p.k), U=p.U, S=p.S, W=p.W, sol=[]) -> TCandidate:
            return sol if i == 0 else rec(
                max_s := np.argmax([Problem(S=list(S), W=W).cost([s]) for s in range(len(S))]),
                i-1,
                np.setdiff1d(U, S[max_s]),
                np.delete(S, max_s),
                np.delete(W, max_s),
                sol + [Utils.where(p.S, S[max_s])])

        return s.eval(p, rec())






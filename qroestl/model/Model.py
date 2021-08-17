from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
from datetime import datetime
from functools import reduce
from typing import List, TypeVar, Generic, Optional, Callable

from qiskit_optimization import QuadraticProgram

from qroestl.utils import Utils


TCandidate = TypeVar('Solution Candidate', bound='Solution')


class Problem(Generic[TCandidate], ABC):
    name: Optional[str] = None

    @abstractmethod
    def feasible(self, c: TCandidate) -> bool:
        raise NotImplementedError

    @abstractmethod
    def cost(self, c: TCandidate) -> float:
        raise NotImplementedError

    def better(self, c1: TCandidate, c2: TCandidate) -> bool:
        return self.cost(c1) < self.cost(c2) if c1 and c2 else (True if c2 else False)

    def superior(self, c1: TCandidate, c2: TCandidate) -> bool:
        return self.feasible(c2) and self.better(c1, c2)

    def all_solutions(self) -> List[TCandidate]:
        return []


TProblem = TypeVar('Problem', bound='Problem')


@dataclass
class Solution(Generic[TCandidate, TProblem], ABC):
    best: (Optional[TCandidate], Optional[float]) = (None, None)
    took: Optional[int] = None

    def eval(self, p: TProblem, c: TCandidate) -> Solution[TCandidate, TProblem]:
        return replace(self, best=(c, p.cost(c)) if p.superior(self.best[0], c) else self.best)

    def __str__(self):
        return f'{self.best[0]} -> {self.best[1]} | {self.took}'


@dataclass
class Solver(Generic[TCandidate, TProblem], ABC):
    pre: List[Callable] = field(default_factory=lambda: [Utils.id])
    post: List[Callable] = field(default_factory=lambda: [Utils.id])
    name: str = None

    def solve_(self, p: TProblem, c: Solution[TCandidate]) -> Solution[TCandidate]:
        raise NotImplementedError

    def solve(self, p: Problem) -> Solution[TCandidate]:
        start = datetime.now()
        solution = self.solve_(p)
        end = datetime.now()
        return replace(solution, took=end - start)

    def __str__(self):
        return self.name


class QPConvertible:
    @abstractmethod
    def to_qp(self) -> QuadraticProgram:
        raise NotImplementedError


class QuboConvertible:
    @abstractmethod
    def to_qubo(self) -> QuadraticProgram:
        raise NotImplementedError


class OperatorConvertible:
    @abstractmethod
    def to_op(self) -> "Operator":
        raise NotImplementedError


@dataclass
class BruteForce(Generic[TCandidate, TProblem], Solver[TCandidate, TProblem]):
    name: str = "BruteForce"

    def solve_(self, p: Problem, s: Solution[TCandidate] = Solution[TCandidate, TProblem]()) -> Solution[TCandidate, TProblem]:
        return reduce(lambda s, candidate: s.eval(p, candidate), p.all_solutions(), s)

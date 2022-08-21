from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
from datetime import datetime
from functools import reduce
from typing import List, TypeVar, Generic, Optional, Tuple

TCandidate = TypeVar('Solution Candidate', bound='Solution')
TProblem = TypeVar('Problem', bound='Problem')


@dataclass
class Approach(Generic[TProblem], ABC):
    name: str


class Problem(Generic[TCandidate], ABC):
    name: str = None

    @abstractmethod
    def feasible(self, c: TCandidate) -> bool:
        raise NotImplementedError

    @abstractmethod
    def value(self, c: TCandidate) -> float:
        raise NotImplementedError

    def better(self, c1: TCandidate, c2: TCandidate) -> bool:
        return self.value(c1) < self.value(c2) if c1 and c2 else (True if c2 else False)

    def superior(self, c1: TCandidate, c2: TCandidate) -> bool:
        return self.feasible(c2) and self.better(c1, c2)

    def all_solutions(self) -> List[TCandidate]:
        return []



@dataclass
class Solution(Generic[TCandidate, TProblem], ABC):
    best: (Optional[TCandidate], Optional[float]) = (None, None)
    wall_clock: Optional[int] = None
    opt_clock: Optional[int] = None
    optimizer_name: str = ''

    def eval(self, p: TProblem, c: TCandidate) -> Solution[TCandidate, TProblem]:
        return replace(self, best=(c, p.value(c)) if p.superior(self.best[0], c) else self.best)

    def to_list(self) -> str:
        return [self.optimizer_name, self.best[1], self.wall_clock, self.opt_clock, self.best[0]]

    def __str__(self) -> str:
        return f'{self.optimizer_name : <20} obj: {self.best[1]} | wall clock: {self.wall_clock} | opt clock: {self.opt_clock} | sol: {(self.best[0])}'


class Converter:
    def convert(self, p, a):
        return p


@dataclass
class Optimizer(Generic[TProblem, TCandidate], ABC):
    name: str
    converter: Converter = Converter()
    kwargs: "kwargs" = field(default_factory=lambda: {})

    def optimize(self, p: TProblem, a: Optional[Approach] = None, c: Solution[TCandidate] = Solution[TCandidate, TProblem]()) -> Solution[TCandidate]:
        start = datetime.now()
        p_conv = self.converter.convert(p, a)
        print(f'Starting actual optimization at {start}')
        solution, opt_clock = self.optimize_(p, p_conv, a, c)
        end = datetime.now()
        return replace(solution, wall_clock=(wc:=end - start), opt_clock=opt_clock if opt_clock else wc, optimizer_name=self.name)

    @abstractmethod
    def optimize_(self, p, p_conv, a, c) -> Tuple[Solution, int]:
        raise NotImplementedError

    def __str__(self):
        return self.name


@dataclass
class BruteForce(Generic[TCandidate, TProblem], Optimizer[TCandidate, TProblem]):
    name: str = "BruteForce"

    def optimize_(self, p: Problem, p_conv, a: Optional[Approach] = None, s: Solution[TCandidate] = Solution[TCandidate, TProblem]()) -> Solution[TCandidate, TProblem]:
        return reduce(lambda s, candidate: s.eval(p, candidate), p.all_solutions(), s), None

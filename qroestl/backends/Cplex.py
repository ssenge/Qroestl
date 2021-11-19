from abc import abstractmethod
from dataclasses import field, dataclass
from typing import TypeVar, Generic

import numpy as np
from docplex.mp.model import Model

from qroestl.model import Approach, Solution, Optimizer, Converter
from qroestl.utils import Utils

TCandidate = TypeVar('Solution Candidate', bound='Solution')
TProblem = TypeVar('Problem', bound='Problem')


class CplexConverter(Converter):
    def convert(self, p, a):
        if isinstance(a, CplexModelConvertible):
            return a.to_cplex_model(p)
        else:
            raise ValueError('Approach not convertible to Cplex Model')

@dataclass
class Optimizer(Generic[TCandidate, TProblem], Optimizer[TCandidate, TProblem]):
    name: str = "Cplex-Local"
    kwargs: "kwargs" = field(default_factory=lambda: {})
    converter: Converter = CplexConverter()

    def optimize_(self, p: TProblem, p_conv, a: Approach, s=Solution[TCandidate, TProblem]()) -> Solution[TCandidate, TProblem]:
        res = [] if not (sol := p_conv.solve()) else sol.get_all_values()
        return s.eval(p, Utils.bits2idx(len(p.S))(np.clip(np.rint(res), 0, 1))), None


class CplexModelConvertible(Generic[TCandidate, TProblem]):
    @abstractmethod
    def to_qiskit_qp(self, p: TProblem) -> Model:
        raise NotImplementedError


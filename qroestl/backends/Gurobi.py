from abc import abstractmethod
from dataclasses import field, dataclass
from datetime import timedelta
from typing import TypeVar, Generic

import gurobipy
import numpy as np
from gurobipy import GRB

from qroestl.model import Approach, Solution, Optimizer, Converter, Model
from qroestl.utils import Utils

TCandidate = TypeVar('Solution Candidate', bound='Solution')
TProblem = TypeVar('Problem', bound='Problem')


class GurobiModelConvertible(Generic[TCandidate, TProblem]):
    @abstractmethod
    def to_gurobi_model(self, p: TProblem) -> gurobipy.Model:
        raise NotImplementedError


class Converter(Converter):
    def convert(self, p, a):
        if isinstance(a, GurobiModelConvertible):
            return a.to_gurobi_model(p)
        else:
            raise ValueError('Approach not convertible to Cplex Model')


@dataclass
class Optimizer(Generic[TCandidate, TProblem], Model.Optimizer[TCandidate, TProblem]):
    name: str = "Gurobi-Local"
    kwargs: "kwargs" = field(default_factory=lambda: {})
    converter: Converter = Converter()

    def optimize_(self, p: TProblem, p_conv, a: Approach, s=Solution[TCandidate, TProblem]()) -> Solution[TCandidate, TProblem]:
        #p_conv.setParam('OutputFlag', 0)
        p_conv.optimize()
        res = [] if p_conv.status is not GRB.OPTIMAL else [v.x for v in p_conv.getVars()]
        return s.eval(p, Utils.bits2idx(len(p.S))(np.clip(np.rint(res), 0, 1))), timedelta(seconds=p_conv.Runtime)


